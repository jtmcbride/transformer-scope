import type { StorySelection } from './continuation';
import { selectionEdits } from './continuation';
import { branchProbabilities } from './branch-probabilities';
import { forkEdits, type ForkPredict } from './story-fork';
import { validateTokens, validateInterventions, requireInput, type ModelConfig, type HeadScaleIntervention } from './engine/contracts';
export interface HeadScanRow { edit: HeadScaleIntervention; before: number; after: number; delta: number; rank: number; choice: number }
export type ScanOrder = 'absolute' | 'increase' | 'decrease';
export function rankHeadScan(rows: HeadScanRow[], order: ScanOrder): HeadScanRow[] {
  const score = (r: HeadScanRow) => order === 'increase' ? r.delta : order === 'decrease' ? -r.delta : Math.abs(r.delta);
  return [...rows].sort((a,b) => score(b)-score(a) || a.edit.layer-b.edit.layer || a.edit.head-b.edit.head || a.edit.factor-b.edit.factor);
}
/** Each trial starts from captured settings, never from the preceding trial's edit. */
export async function scanStoryHeads(source: StorySelection, config: ModelConfig, predict: ForkPredict, emit: (row: HeadScanRow) => void, cancelled: () => boolean): Promise<void> {
  validateTokens(source.prefix, config); validateInterventions(selectionEdits(source), config, source.prefix.length);
  if (cancelled()) return;
  const before = await predict([...source.prefix], selectionEdits(source));
  if (cancelled()) return;
  const reference = branchProbabilities(before, before, source.choice.tokenId, source.settings, source.step);
  requireInput(reference.beforeChoice === source.choice.tokenId && Math.abs(reference.selected.before - source.choice.probability) <= 1e-6, 'The scan could not reproduce the source prediction.');
  for (let layer=0; layer<config.layers; layer++) for (let head=0; head<config.heads; head++) for (const factor of [0, 1, 2]) {
    if (cancelled()) return;
    const edit: HeadScaleIntervention = { id: 'head-scan', kind: 'head.scale', layer, head, factor, tokens: { kind: 'all' } };
    const edits = forkEdits({ source, edit, duration: 'prediction', maxTokens: 1 }, true, 0);
    validateInterventions(edits, config, source.prefix.length);
    const after = await predict([...source.prefix], edits);
    if (cancelled()) return;
    const result = branchProbabilities(before, after, source.choice.tokenId, source.settings, source.step);
    emit({ edit, before: result.selected.before, after: result.selected.after, delta: result.selected.delta, rank: result.selected.afterRank, choice: result.afterChoice });
  }
}
