import type { StorySelection } from './continuation';
import { branchProbabilities, headPairInteraction } from './branch-probabilities';
import { forkEdits, type ForkPredict } from './story-fork';
import { storyScanReference, type ScanOrder } from './story-head-scan';
import { requireInput, validateInterventions, type HeadScaleIntervention, type ModelConfig } from './engine/contracts';
export const pairStrengthOptions = [
  { value: 'mute', label: 'Mute both (0× + 0×)', factors: [[0, 0]] },
  { value: 'amplify', label: 'Amplify both (2× + 2×)', factors: [[2, 2]] },
  { value: 'opposite', label: 'Mute + Amplify (both assignments)', factors: [[0, 2], [2, 0]] },
  { value: 'normal', label: 'Normal both (1× + 1×)', factors: [[1, 1]] },
  { value: 'mute-normal', label: 'Mute + Normal (both assignments)', factors: [[0, 1], [1, 0]] },
  { value: 'normal-amplify', label: 'Normal + Amplify (both assignments)', factors: [[1, 2], [2, 1]] },
  { value: 'all', label: 'All strengths (0×, 1×, 2× on each head)', factors: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]] },
] as const;
export type PairStrengths = typeof pairStrengthOptions[number]['value'];
export type HeadPair = [HeadScaleIntervention, HeadScaleIntervention];
export interface HeadPairScanRow { edits: HeadPair; before: number; after: number; delta: number; rank: number; choice: number; individualProbabilities: [number, number]; sumDelta: number; interaction: number }
export type PairScanOrder = ScanOrder | 'interaction-absolute' | 'interaction-positive' | 'interaction-negative';
function factorsFor(strengths: PairStrengths) {
  const option = pairStrengthOptions.find(item => item.value === strengths);
  requireInput(!!option, 'Choose valid pair strengths.');
  return option!.factors;
}
export function pairScanTotal(config: ModelConfig, strengths: PairStrengths): number {
  const heads = config.layers * config.heads;
  return heads * (heads - 1) / 2 * factorsFor(strengths).length;
}
/** Canonical unordered head pairs; mixed strengths include both assignments exactly once. */
export function* headPairTrials(config: ModelConfig, strengths: PairStrengths): Generator<HeadPair> {
  const factors = factorsFor(strengths), heads = config.layers * config.heads;
  const edit = (index: number, factor: number): HeadScaleIntervention => ({ id: 'head-scan', kind: 'head.scale', layer: Math.floor(index / config.heads), head: index % config.heads, factor, tokens: { kind: 'all' } });
  for (let a = 0; a < heads; a++) for (let b = a + 1; b < heads; b++) for (const [first, second] of factors) yield [edit(a, first), edit(b, second)];
}
export function rankHeadPairScan(rows: HeadPairScanRow[], order: PairScanOrder): HeadPairScanRow[] {
  const score = (r: HeadPairScanRow) => order === 'interaction-absolute' ? Math.abs(r.interaction) : order === 'interaction-positive' ? r.interaction : order === 'interaction-negative' ? -r.interaction : order === 'increase' ? r.delta : order === 'decrease' ? -r.delta : Math.abs(r.delta);
  return [...rows].sort((a, b) => score(b) - score(a) || a.edits[0].layer - b.edits[0].layer || a.edits[0].head - b.edits[0].head || a.edits[1].layer - b.edits[1].layer || a.edits[1].head - b.edits[1].head || a.edits[0].factor - b.edits[0].factor || a.edits[1].factor - b.edits[1].factor);
}
/** Measure the actual joint distribution; each trial independently replaces only its two heads. */
export async function scanStoryHeadPairs(source: StorySelection, config: ModelConfig, strengths: PairStrengths, predict: ForkPredict, emit: (row: HeadPairScanRow) => void, cancelled: () => boolean): Promise<void> {
  factorsFor(strengths);
  const before = await storyScanReference(source, config, predict, cancelled);
  if (!before) return;
  const singles = new Map<string, number>();
  async function individual(edit: HeadScaleIntervention): Promise<number | null> {
    const key = `${edit.layer}:${edit.head}:${edit.factor}`;
    const saved = singles.get(key);
    if (saved !== undefined) return saved;
    const applied = forkEdits({ source, edit, duration: 'prediction', maxTokens: 1 }, true, 0);
    validateInterventions(applied, config, source.prefix.length);
    const logits = await predict([...source.prefix], applied);
    if (cancelled()) return null;
    const probability = branchProbabilities(before!, logits, source.choice.tokenId, source.settings, source.step).selected.after;
    singles.set(key, probability);
    return probability;
  }
  for (const edits of headPairTrials(config, strengths)) {
    if (cancelled()) return;
    const applied = forkEdits({ source, edit: edits[0], additionalEdits: [edits[1]], duration: 'prediction', maxTokens: 1 }, true, 0);
    validateInterventions(applied, config, source.prefix.length);
    const after = await predict([...source.prefix], applied);
    if (cancelled()) return;
    const result = branchProbabilities(before, after, source.choice.tokenId, source.settings, source.step);
    const first = await individual(edits[0]);
    if (first === null || cancelled()) return;
    const second = await individual(edits[1]);
    if (second === null || cancelled()) return;
    const individualProbabilities: [number, number] = [first, second];
    emit({ edits, before: result.selected.before, after: result.selected.after, delta: result.selected.delta, rank: result.selected.afterRank, choice: result.afterChoice,
      individualProbabilities, ...headPairInteraction(result.selected.before, result.selected.after, individualProbabilities) });
  }
}
