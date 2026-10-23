import { chooseToken, drawAt, type StoryFrame, type StorySelection } from './continuation';
import { requireInput, type ModelConfig } from './engine/contracts';
import { logSoftmax } from './engine/comparison';
import { forkEdits, proposedEdits, selectForkToken, type ForkPlan, type ForkPredict } from './story-fork';
import { SharedStoryFork } from './story-shared-fork';
import { headPairInteraction } from './branch-probabilities';
import type { WithSource } from './story-heatmap-comparison';

export interface InteractionPoint {
  index: number; step: number; tokenId: number; recordedProbability: number; active: boolean;
  targetTokenId: number; alternatives: {tokenId:number;probability:number}[];
  before: number; after: number; individualProbabilities: [number,number]; delta: number; sumDelta: number; interaction: number;
}
export type InteractionTracks = [InteractionPoint[],InteractionPoint[]];
export type InteractionTarget = {kind:'generated'} | {kind:'fixed';tokenId:number};
export type InteractionTrend = 'Starting point' | 'Near zero' | 'Growing' | 'Fading' | 'Fading to zero' | 'Reversed' | 'Steady';
/** Trend labels use the same four-decimal percentage-point precision as the display, never raw round-off. */
export function interactionTrend(point: InteractionPoint, previous?: InteractionPoint): InteractionTrend {
  const current = Number((point.interaction*100).toFixed(4));
  if (!previous || previous.index !== point.index-1) return current === 0 ? 'Near zero' : 'Starting point';
  const before = Number((previous.interaction*100).toFixed(4));
  if (current === 0) return before === 0 ? 'Near zero' : 'Fading to zero';
  if (before*current < 0) return 'Reversed';
  return Math.abs(current) > Math.abs(before) ? 'Growing' : Math.abs(current) < Math.abs(before) ? 'Fading' : 'Steady';
}

/** Replay the actual path, then measure counterfactual single/pair edits on its exact pre-token prefix. */
async function measurePoint(plan: ForkPlan, frame: StoryFrame, branch: StorySelection['branch'], index: number,
  predict: ForkPredict, cancelled: () => boolean, target:InteractionTarget): Promise<InteractionPoint | null> {
  const source = selectForkToken(plan,frame,branch,index), active = plan.duration === 'continuation' || index === 0;
  const cache = new Map<string,ArrayLike<number>>();
  async function logits(edits: ReturnType<typeof forkEdits>) {
    const key = JSON.stringify(edits);
    if (!cache.has(key)) cache.set(key,await predict([...source.prefix],edits));
    return cache.get(key)!;
  }
  const base = await logits(forkEdits(plan,false,index)); if (cancelled()) return null;
  const joint = await logits(forkEdits(plan,true,index)); if (cancelled()) return null;
  const replay = chooseToken(branch === 'original' ? base : joint,source.settings,drawAt(source.settings.seed,source.step-1));
  requireInput(replay.tokenId === source.choice.tokenId && Math.abs(replay.probability-source.choice.probability) <= 1e-6, 'This trajectory token could not be reproduced.');
  const targetTokenId=target.kind==='fixed'?target.tokenId:source.choice.tokenId;
  const actual=logSoftmax(branch==='original'?base:joint);
  const alternatives=Array.from(actual,(value,tokenId)=>({tokenId,probability:Math.exp(value)}))
    .filter(row=>row.tokenId!==0&&row.tokenId!==2&&row.tokenId!==source.choice.tokenId)
    .sort((a,b)=>b.probability-a.probability||a.tokenId-b.tokenId).slice(0,5);
  const probability = (values: ArrayLike<number>) => Math.exp(logSoftmax(values)[targetTokenId]);
  const before = probability(base), after = probability(joint), individuals: number[] = [];
  for (const edit of proposedEdits(plan)) {
    const values = await logits(forkEdits({...plan,edit,additionalEdits:undefined},true,index));
    if (cancelled()) return null;
    individuals.push(probability(values));
  }
  return {index,step:source.step,tokenId:source.choice.tokenId,targetTokenId,alternatives,recordedProbability:source.choice.probability,active,
    before,after,individualProbabilities:individuals as [number,number],delta:after-before,...headPairInteraction(before,after,individuals)};
}

/** Sources advance by generation offset; ended paths have no invented or zero-filled tokens. */
export async function traceStoryInteractions(plans: [ForkPlan,ForkPlan], frames: [StoryFrame,StoryFrame], branch: StorySelection['branch'], config: ModelConfig,
  withSource: WithSource, emit: (side: 0|1, point: InteractionPoint) => void, cancelled: () => boolean, target:InteractionTarget={kind:'generated'}): Promise<void> {
  const capturedTarget=structuredClone(target);
  requireInput(capturedTarget?.kind==='generated'||capturedTarget?.kind==='fixed'&&Number.isInteger(capturedTarget.tokenId)
    &&capturedTarget.tokenId>=1&&capturedTarget.tokenId!==2&&capturedTarget.tokenId<config.vocabularySize,'Choose an eligible next token from this model.');
  const shared = new SharedStoryFork(plans,config); shared.restore(frames);
  const captured = {plans:shared.plans,frames:shared.frames};
  const count = Math.max(...captured.frames.map(frame => frame[branch]!.choices.length));
  for (let index=0; index<count; index++) for (const side of [0,1] as const) {
    if (cancelled()) return;
    const plan = captured.plans[side], frame = captured.frames[side];
    if (index >= frame[branch]!.choices.length) continue;
    const source = selectForkToken(plan,frame,branch,index);
    await withSource(source,async predict => {
      if (cancelled()) return;
      const point = await measurePoint(plan,frame,branch,index,predict,cancelled,capturedTarget);
      if (point && !cancelled()) emit(side,point);
    });
  }
}
