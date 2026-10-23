import { branchProbabilities, headPairInteraction, type BranchProbabilities } from './branch-probabilities';
import { forkEdits, proposedEdits, type ForkPlan, type ForkPredict } from './story-fork';
import { requireInput, type HeadScaleIntervention } from './engine/contracts';
export interface StoryProbabilities extends BranchProbabilities {
  individuals: { edit: HeadScaleIntervention; probability: number; delta: number }[];
  interaction: number;
}
/** Compare each proposed edit and the pair on the exact source prefix; never add logits or probabilities. */
export async function measureStoryProbabilities(plan: ForkPlan, predict: ForkPredict, cancelled = () => false, targetTokenId = plan.source.choice.tokenId): Promise<StoryProbabilities | null> {
  const edits = proposedEdits(plan);
  const before = await predict(plan.source.prefix, forkEdits(plan, false, 0));
  if (cancelled()) return null;
  const reference = branchProbabilities(before, before, plan.source.choice.tokenId, plan.source.settings, plan.source.step);
  if (reference.beforeChoice !== plan.source.choice.tokenId || Math.abs(reference.selected.before - plan.source.choice.probability) > 1e-6) throw new Error('The source prediction could not be reproduced.');
  requireInput(Number.isInteger(targetTokenId) && targetTokenId > 0 && targetTokenId < before.length && targetTokenId !== 2, 'Choose a valid output token.');
  const after = await predict(plan.source.prefix, forkEdits(plan, true, 0));
  if (cancelled()) return null;
  const report = branchProbabilities(before, after, targetTokenId, plan.source.settings, plan.source.step);
  const individuals: StoryProbabilities['individuals'] = [];
  if (edits.length === 2) for (const edit of edits) {
    const alone = await predict(plan.source.prefix, forkEdits({ ...plan, edit, additionalEdits: undefined }, true, 0));
    if (cancelled()) return null;
    const row = branchProbabilities(before, alone, targetTokenId, plan.source.settings, plan.source.step).selected;
    individuals.push({ edit: structuredClone(edit), probability: row.after, delta: row.delta });
  }
  return { ...report, individuals, interaction: individuals.length === 2 ? headPairInteraction(report.selected.before, report.selected.after, individuals.map(row => row.probability)).interaction : 0 };
}
