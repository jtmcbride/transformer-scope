import { StoryContinuation, type StorySelection, type StoryFrame } from './continuation';
import { requireInput, validateInterventions, validateTokens, type HeadScaleIntervention, type ModelConfig } from './engine/contracts';

export type ForkDuration = 'continuation' | 'prediction';
export interface ForkPlan { source: StorySelection; edit: HeadScaleIntervention; duration: ForkDuration; maxTokens: number }
export type ForkPredict = (tokens: number[], edits: HeadScaleIntervention[]) => Promise<ArrayLike<number>>;

/** Replace a captured edit on the same head; retain edits on other heads. Never multiply overlapping edits. */
export function forkEdits(plan: ForkPlan, changed: boolean, step: number): HeadScaleIntervention[] {
  const inherited = plan.source.edit ? [plan.source.edit] : [];
  if (!changed || plan.duration === 'prediction' && step > 0) return structuredClone(inherited);
  return structuredClone([...inherited.filter(edit => edit.layer !== plan.edit.layer || edit.head !== plan.edit.head), plan.edit]);
}

/** Both continuations start before the selected output, with the original sampling counter. */
export class StoryFork {
  readonly plan: ForkPlan;
  private session: StoryContinuation;
  constructor(plan: ForkPlan, config: ModelConfig) {
    validateTokens(plan.source.prefix, config);
    requireInput(plan.duration === 'continuation' || plan.duration === 'prediction', 'Choose how long the edit applies.');
    requireInput(Number.isInteger(plan.source.step) && plan.source.step >= 1 && plan.source.step <= 64, 'Invalid source step.');
    requireInput(plan.edit.tokens.kind === 'all' && (!plan.source.edit || plan.source.edit.tokens.kind === 'all'), 'Forks use all-position head edits.');
    validateInterventions(forkEdits(plan, false, 0), config, plan.source.prefix.length);
    validateInterventions(forkEdits(plan, true, 0), config, plan.source.prefix.length);
    this.plan = structuredClone(plan);
    this.session = new StoryContinuation(plan.source.prefix, { ...plan.source.settings, maxTokens: plan.maxTokens }, config.contextLength, plan.edit, plan.source.step - 1);
  }
  get frame(): StoryFrame { return this.session.frame; }
  get finished(): boolean { return this.session.finished; }
  step(predict: ForkPredict): Promise<StoryFrame> {
    const step = this.frame.step;
    return this.session.step((tokens, edit) => predict(tokens, forkEdits(this.plan, !!edit, step)));
  }
}

/** Token-step alignment, not word alignment. An ended branch differs from an additional token. */
export function firstForkDifference(frame: StoryFrame): number | null {
  if (!frame.edited) return null;
  const count = Math.max(frame.original.choices.length, frame.edited.choices.length);
  for (let i = 0; i < count; i++) if (frame.original.choices[i]?.tokenId !== frame.edited.choices[i]?.tokenId) return i;
  return null;
}
