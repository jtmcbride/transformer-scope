import { chooseToken, drawAt, type Sampling } from './continuation';
import { logSoftmax } from './engine/comparison';
import { requireInput } from './engine/contracts';
export interface BranchProbabilityRow { tokenId: number; before: number; after: number; delta: number; beforeRank: number; afterRank: number }
export interface BranchProbabilities { selected: BranchProbabilityRow; rows: BranchProbabilityRow[]; beforeChoice: number; afterChoice: number }
/** Non-additivity in raw probability, measured against one shared source distribution. */
export function headPairInteraction(before: number, after: number, individuals: readonly number[]): { sumDelta: number; interaction: number } {
  requireInput(individuals.length === 2 && [before, after, ...individuals].every(p => Number.isFinite(p) && p >= 0 && p <= 1), 'Use two individual probabilities and a shared source probability.');
  const sumDelta = individuals.reduce((sum, probability) => sum + (probability - before), 0);
  return { sumDelta, interaction: (after - before) - sumDelta };
}
/** Both logits describe the same exact prefix. BOS/PAD stay in softmax but cannot be generated. */
export function branchProbabilities(before: ArrayLike<number>, after: ArrayLike<number>, selected: number, settings: Sampling, step: number): BranchProbabilities {
  requireInput(before.length === after.length && Number.isInteger(selected) && selected >= 0 && selected < before.length, 'Invalid branch probability comparison.');
  const a = logSoftmax(before), b = logSoftmax(after);
  const rank = (logits: ArrayLike<number>) => Array.from({ length: logits.length }, (_, id) => id).filter(id => id !== 0 && id !== 2).sort((x, y) => logits[y] - logits[x] || x - y);
  const ar = rank(before), br = rank(after), ai = new Map(ar.map((id, i) => [id, i + 1])), bi = new Map(br.map((id, i) => [id, i + 1]));
  const draw = drawAt(settings.seed, step - 1), beforeChoice = chooseToken(before, settings, draw).tokenId, afterChoice = chooseToken(after, settings, draw).tokenId;
  const row = (tokenId: number): BranchProbabilityRow => ({ tokenId, before: Math.exp(a[tokenId]), after: Math.exp(b[tokenId]), delta: Math.exp(b[tokenId]) - Math.exp(a[tokenId]), beforeRank: ai.get(tokenId) ?? 0, afterRank: bi.get(tokenId) ?? 0 });
  const ids = new Set([selected, beforeChoice, afterChoice, ...ar.slice(0, 5), ...br.slice(0, 5)]);
  return { selected: row(selected), rows: [...ids].map(row).sort((x, y) => Math.max(y.before, y.after) - Math.max(x.before, x.after) || x.tokenId - y.tokenId), beforeChoice, afterChoice };
}
