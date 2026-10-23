import type { StoryFrame } from './continuation';
import { requireInput, type ModelConfig } from './engine/contracts';
import { proposedEdits, StoryFork, type ForkPlan, type ForkPredict } from './story-fork';

/** Two source-specific references and edits, with one atomic four-path generation step. */
export class SharedStoryFork {
  readonly plans: [ForkPlan, ForkPlan];
  private current: [StoryFrame, StoryFrame];
  private busy = false;
  constructor(plans: [ForkPlan, ForkPlan], private config: ModelConfig) {
    const forks = plans.map(plan => new StoryFork(plan, config));
    requireInput(plans.every(plan => proposedEdits(plan).length === 2), 'Choose a pair of heads for both stories.');
    requireInput(JSON.stringify(proposedEdits(plans[0])) === JSON.stringify(proposedEdits(plans[1]))
      && plans[0].duration === plans[1].duration && plans[0].maxTokens === plans[1].maxTokens, 'Both stories must use the same head changes, duration, and token limit.');
    this.plans = structuredClone(plans);
    this.current = forks.map(fork => fork.frame) as [StoryFrame, StoryFrame];
  }
  get frames(): [StoryFrame, StoryFrame] { return structuredClone(this.current); }
  get finished(): boolean { return this.current.every(frame => !!frame.original.finish && !!frame.edited?.finish); }
  restore(frames: [StoryFrame, StoryFrame]): void {
    const forks = this.plans.map((plan, i) => { const fork = new StoryFork(plan, this.config); fork.restore(frames[i]); return fork; });
    const step = Math.max(...frames.map(frame => frame.step));
    requireInput(forks.every(fork => fork.finished || fork.frame.step === step), 'Shared story comparisons must pause at the same generation step.');
    this.current = structuredClone(frames);
  }
  async step(predict: ForkPredict, cancelled: () => boolean = () => false): Promise<[StoryFrame, StoryFrame] | null> {
    requireInput(!this.busy, 'A shared story step is already running.');
    if (cancelled()) return null;
    if (this.finished) return this.frames;
    this.busy = true;
    try {
      const next = [] as unknown as [StoryFrame, StoryFrame];
      for (const side of [0, 1] as const) {
        if (cancelled()) return null;
        const fork = new StoryFork(this.plans[side], this.config);
        fork.restore(this.current[side]);
        next[side] = await fork.step(predict);
        if (this.current[side].step === 0 && next[side].original.choices.length) {
          const replay = next[side].original.choices[0], recorded = this.plans[side].source.choice;
          requireInput(replay.tokenId === recorded.tokenId && Math.abs(replay.probability - recorded.probability) <= 1e-6, 'The shared continuation did not reproduce its recorded source token.');
        }
      }
      if (cancelled()) return null;
      this.current = next;
      return this.frames;
    } finally { this.busy = false; }
  }
}
