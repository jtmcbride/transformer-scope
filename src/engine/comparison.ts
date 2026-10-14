import { requireInput, type ModelTokenizer } from './contracts';

export interface TokenProbability { tokenId: number; text: string; logit: number; probability: number }
export interface TokenDelta extends TokenProbability { baselineProbability: number; probabilityDelta: number }
export interface PredictionComparison {
  baselineTopTokens: TokenProbability[]; modifiedTopTokens: TokenProbability[]; mostChangedTokens: TokenDelta[];
  displayTokens: TokenDelta[];
  metrics: { jsDivergence: number; maxAbsoluteLogitChange: number; targetLogProbabilityDelta?: number };
}
export function logSoftmax(values: ArrayLike<number>): Float64Array {
  requireInput(values.length > 0, 'Cannot compare an empty prediction.');
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    requireInput(Number.isFinite(values[i]), 'Prediction contains a non-finite value.');
    max = Math.max(max, values[i]);
  }
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += Math.exp(values[i] - max);
  return Float64Array.from(values, value => (value - max) - Math.log(sum));
}
export function comparePredictions(baseline: ArrayLike<number>, modified: ArrayLike<number>, tokenizer: ModelTokenizer,
  topK = 8, targetTokenId?: number): PredictionComparison {
  requireInput(baseline.length === modified.length && baseline.length === tokenizer.tokens.length, 'Prediction vocabulary mismatch.');
  requireInput(Number.isInteger(topK) && topK > 0 && topK <= baseline.length, 'Invalid prediction display limit.');
  requireInput(targetTokenId === undefined || (Number.isInteger(targetTokenId) && targetTokenId >= 0 && targetTokenId < baseline.length), 'Target token is outside the vocabulary.');
  const lb = logSoftmax(baseline), lm = logSoftmax(modified);
  const original: TokenProbability[] = [], changed: TokenProbability[] = [], deltas: TokenDelta[] = [];
  let jsDivergence = 0, maxAbsoluteLogitChange = 0;
  for (let i = 0; i < baseline.length; i++) {
    const pb = Math.exp(lb[i]), pm = Math.exp(lm[i]);
    const larger = Math.max(lb[i], lm[i]);
    const logMixture = larger + (Math.log(Math.exp(lb[i] - larger) + Math.exp(lm[i] - larger)) - Math.LN2);
    jsDivergence += .5 * (pb * (lb[i] - logMixture) + pm * (lm[i] - logMixture));
    maxAbsoluteLogitChange = Math.max(maxAbsoluteLogitChange, Math.abs(modified[i] - baseline[i]));
    const record = { tokenId: i, text: tokenizer.tokens[i], logit: modified[i], probability: pm };
    original.push({ ...record, logit: baseline[i], probability: pb }); changed.push(record);
    deltas.push({ ...record, baselineProbability: pb, probabilityDelta: pm - pb });
  }
  const rank = (a: TokenProbability, b: TokenProbability) => b.probability - a.probability || a.tokenId - b.tokenId;
  const baselineTopTokens = original.sort(rank).slice(0, topK);
  const modifiedTopTokens = changed.sort(rank).slice(0, topK);
  const mostChangedTokens = [...deltas].sort((a, b) => Math.abs(b.probabilityDelta) - Math.abs(a.probabilityDelta) || a.tokenId - b.tokenId).slice(0, topK);
  const visible = new Set([...baselineTopTokens, ...modifiedTopTokens, ...mostChangedTokens].map(token => token.tokenId));
  if (targetTokenId !== undefined) visible.add(targetTokenId);
  const displayTokens = deltas.filter(token => visible.has(token.tokenId)).sort((a, b) =>
    Math.max(b.probability, b.baselineProbability) - Math.max(a.probability, a.baselineProbability) || a.tokenId - b.tokenId);
  return {
    baselineTopTokens, modifiedTopTokens, mostChangedTokens, displayTokens,
    metrics: { jsDivergence: Math.max(0, jsDivergence), maxAbsoluteLogitChange,
      ...(targetTokenId === undefined ? {} : { targetLogProbabilityDelta: lm[targetTokenId] - lb[targetTokenId] }) },
  };
}
