import type { BpeDocument } from './bpe';
export { tokenize, textForTokens, tokenLabel } from './tokenizer';
export interface ModelConfig {
  layers: number;
  heads: number;
  modelDim: number;
  mlpDim: number;
  vocabularySize: number;
  contextLength: number;
  layerNormEpsilon: number;
}

export interface TensorRecord { shape: number[]; values: number[] }
export interface TensorEntry { name: string; shape: number[]; byteOffset: number; byteLength: number }
export interface ModelManifest {
  schemaVersion: 1;
  architectureVersion: 'preln-absolute-tied-v1';
  modelId: string;
  modelRevision: string;
  config: ModelConfig;
  dtype: 'float32-le';
  matrixOrientation: 'input-output';
  conventions: {
    gelu: 'tanh'; layerNormVariance: 'population'; position: 'learned-absolute';
    tiedEmbeddings: true; dropout: 0; maskValue: -1e9;
  };
  weightsFile: string;
  tokenizerFile: string;
  weightsSha256: string;
  tokenizerSha256: string;
  tensors: TensorEntry[];
  provenance: { purpose: string; seed: number; torchVersion: string; generatorVersion: number };
}
export interface SymbolTokenizer {
  schemaVersion: 1; kind: 'symbols'; bosTokenId: number; tokens: string[];
}
export type ModelTokenizer = SymbolTokenizer | BpeDocument;
export interface ModelArtifact {
  manifest: ModelManifest;
  tokenizer: ModelTokenizer;
  weights: Record<string, TensorRecord>;
}
export type TokenSelection = { kind: 'all' } | { kind: 'indices'; indices: number[] };
export interface HeadScaleIntervention {
  id: string; kind: 'head.scale'; layer: number; head: number; tokens: TokenSelection; factor: number;
}
/** Overwrites one whole residual row: before block `layer` (block.input) or after its attention is added (attention.residual). */
export interface ResidualReplaceIntervention {
  id: string; kind: 'residual.replace'; layer: number; stage: ResidualHook; position: number; values: number[];
}
export type ResidualHook = 'block.input' | 'attention.residual';
export const residualHooks: readonly ResidualHook[] = ['block.input', 'attention.residual'];
/** Replaces one head's post-softmax attention row at one query position with a causal distribution over sources. */
export interface AttentionOverrideIntervention {
  id: string; kind: 'attention.override'; layer: number; head: number; query: number; weights: number[];
}
export type Intervention = HeadScaleIntervention | ResidualReplaceIntervention | AttentionOverrideIntervention;
/** Submitted attention weights are divided by their float64 sum, then rounded to float32 (as in the PyTorch reference). */
export function normalizedAttention(weights: readonly number[]): Float32Array {
  const sum = weights.reduce((total, weight) => total + weight, 0);
  return Float32Array.from(weights, weight => weight / sum);
}
/** Replacement entries are bounded so a submitted vector cannot overflow float32 normalization. */
export const MAX_RESIDUAL_MAGNITUDE = 1e4;
export type ActivationStage =
  | 'embedding.output' | 'block.input' | 'attention.normalized'
  | 'attention.query' | 'attention.key' | 'attention.value'
  | 'attention.scores' | 'attention.maskedScores' | 'attention.weights'
  | 'attention.head.raw' | 'attention.head.modified' | 'attention.projected'
  | 'attention.residual' | 'mlp.output' | 'block.output' | 'final.normalized' | 'logits';
export interface ActivationReference { stage: ActivationStage; layer?: number }
export const stages: readonly ActivationStage[] = [
  'embedding.output', 'block.input', 'attention.normalized', 'attention.query', 'attention.key',
  'attention.value', 'attention.scores', 'attention.maskedScores', 'attention.weights',
  'attention.head.raw', 'attention.head.modified', 'attention.projected', 'attention.residual',
  'mlp.output', 'block.output', 'final.normalized', 'logits',
];
export type EngineErrorCode = 'INVALID_INPUT' | 'INCOMPATIBLE_MODEL' | 'UNSUPPORTED_BACKEND' | 'NUMERICAL_FAILURE' | 'RELEASED_RUN' | 'RESOURCE_EXHAUSTION' | 'STALE_SESSION';
export class EngineError extends Error {
  constructor(public readonly code: EngineErrorCode, message: string) { super(message); this.name = 'EngineError'; }
}
export function requireInput(condition: unknown, message: string): asserts condition {
  if (!condition) throw new EngineError('INVALID_INPUT', message);
}
export function activationKey(reference: ActivationReference): string {
  return reference.layer === undefined ? reference.stage : `${reference.layer}:${reference.stage}`;
}

export function validateTokens(tokens: readonly number[], config: ModelConfig): void {
  requireInput(Array.isArray(tokens) && tokens.length >= 1 && tokens.length <= config.contextLength,
    `Use between 1 and ${config.contextLength} tokens, including the beginning-of-sequence token.`);
  requireInput(tokens.every(t => Number.isInteger(t) && t >= 0 && t < config.vocabularySize), 'Token IDs must be in the model vocabulary.');
}

export function validateInterventions(patches: readonly Intervention[], config: ModelConfig, length: number): void {
  requireInput(Array.isArray(patches), 'Interventions must be an array.');
  const occupied = new Set<string>();
  for (const patch of patches) {
    requireInput(patch && ['head.scale', 'residual.replace', 'attention.override'].includes(patch.kind) && typeof patch.id === 'string', 'Unsupported intervention.');
    if (patch.kind === 'attention.override') {
      requireInput(Number.isInteger(patch.layer) && patch.layer >= 0 && patch.layer < config.layers, 'Layer is outside the model.');
      requireInput(Number.isInteger(patch.head) && patch.head >= 0 && patch.head < config.heads, 'Head is outside the model.');
      requireInput(Number.isInteger(patch.query) && patch.query >= 0 && patch.query < length, 'Query position is outside the input.');
      requireInput(Array.isArray(patch.weights) && patch.weights.length === length, `Attention weights need one value per input position (${length}).`);
      requireInput(patch.weights.every((weight: number) => Number.isFinite(weight) && weight >= 0 && weight <= MAX_RESIDUAL_MAGNITUDE), 'Attention weights must be finite, non-negative, and at most 10000.');
      requireInput(patch.weights.every((weight: number, source: number) => source <= patch.query || weight === 0), 'Attention cannot assign weight to a later position.');
      requireInput(patch.weights.reduce((sum: number, weight: number) => sum + weight, 0) > 0, 'Attention weights must have a positive sum.');
      const key = `attention:${patch.layer}:${patch.head}:${patch.query}`;
      requireInput(!occupied.has(key), 'An attention row can be overridden only once.');
      occupied.add(key);
      continue;
    }
    if (patch.kind === 'residual.replace') {
      requireInput(Number.isInteger(patch.layer) && patch.layer >= 0 && patch.layer < config.layers, 'Layer is outside the model.');
      requireInput(residualHooks.includes(patch.stage), 'Residual replacement must target block.input or attention.residual.');
      requireInput(Number.isInteger(patch.position) && patch.position >= 0 && patch.position < length, 'Token position is outside the input.');
      requireInput(Array.isArray(patch.values) && patch.values.length === config.modelDim &&
        patch.values.every((value: number) => Number.isFinite(value) && Math.abs(value) <= MAX_RESIDUAL_MAGNITUDE),
      `A replacement must contain ${config.modelDim} finite values within ±${MAX_RESIDUAL_MAGNITUDE}.`);
      const key = `residual:${patch.layer}:${patch.stage}:${patch.position}`;
      requireInput(!occupied.has(key), 'Residual replacements cannot target the same layer, stage, and position twice.');
      occupied.add(key);
      continue;
    }
    requireInput(Number.isInteger(patch.layer) && patch.layer >= 0 && patch.layer < config.layers, 'Layer is outside the model.');
    requireInput(Number.isInteger(patch.head) && patch.head >= 0 && patch.head < config.heads, 'Head is outside the model.');
    requireInput(Number.isFinite(patch.factor) && patch.factor >= 0 && patch.factor <= 2, 'Head strength must be between 0 and 2.');
    requireInput(patch.tokens && (patch.tokens.kind === 'all' || patch.tokens.kind === 'indices'), 'Invalid token selection.');
    const positions = patch.tokens.kind === 'all' ? Array.from({ length }, (_, i) => i) : patch.tokens.indices;
    requireInput(Array.isArray(positions), 'Token indices must be an array.');
    for (const position of positions) {
      requireInput(Number.isInteger(position) && position >= 0 && position < length, 'Token position is outside the input.');
      const key = `${patch.layer}:${patch.head}:${position}`;
      requireInput(!occupied.has(key), 'Interventions cannot overlap or contain duplicate token positions.');
      occupied.add(key);
    }
  }
}

export function canReachPrediction(patch: Intervention, predictionPosition: number, config: ModelConfig, length: number): boolean {
  if (patch.kind === 'attention.override')
    return patch.query === predictionPosition || (patch.query < predictionPosition && patch.layer < config.layers - 1);
  if (patch.kind === 'residual.replace') {
    // block.input feeds this layer's attention; attention.residual reaches other positions only through a later layer.
    return patch.position === predictionPosition || (patch.position < predictionPosition && (patch.stage === 'block.input' || patch.layer < config.layers - 1));
  }
  if (patch.factor === 1) return false;
  const positions = patch.tokens.kind === 'all' ? Array.from({ length }, (_, i) => i) : patch.tokens.indices;
  return positions.some(position => position === predictionPosition || (position < predictionPosition && patch.layer < config.layers - 1));
}
