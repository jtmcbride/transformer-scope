import { comparePredictions, type PredictionComparison } from './comparison';
import { requireInput, residualHooks, tokenize, validateInterventions, validateTokens, type Intervention, type ModelManifest, type SymbolTokenizer } from './contracts';

export interface RuntimePrecision { dtype: 'float32'; float32Textures: boolean | null; webglPacking: boolean | null }
/** Exports are written as version 2; version 1 (one head scaling) files remain importable. */
export const EXPERIMENT_SCHEMA_VERSION = 2;
export interface SavedInput { text: string; tokenIds: number[] }
export interface ExperimentDocument {
  format: 'transformer-microscope-experiment'; schemaVersion: 1 | 2; capturedAt: string;
  checkpoint: Pick<ModelManifest, 'modelId' | 'modelRevision' | 'architectureVersion' | 'weightsSha256' | 'tokenizerSha256'>;
  input: { text: string; tokenIds: number[] };
  /** source: the input whose residual a residual.replace copies (version 2 patches only). */
  observation: { predictionPosition: number; targetTokenId: number; interventions: Intervention[]; source?: SavedInput };
  results: { baselineLogits: number[]; modifiedLogits: number[]; metrics: PredictionComparison['metrics']; topK: number;
    temperature: 1; logarithmUnit: 'nats'; backend: 'cpu' | 'webgl'; precision: RuntimePrecision };
}
export interface ReplayReport { matches: boolean; baselineMaximumError: number; modifiedMaximumError: number; savedBackend: string; currentBackend: string }
export const MAX_EXPERIMENT_BYTES = 1024 * 1024;
const hex = /^[a-f0-9]{64}$/u;
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const isVector = (value: unknown): value is number[] => Array.isArray(value) && value.length > 0 && value.length <= 8192 && value.every(Number.isFinite);
const isSavedInput = (value: unknown): value is SavedInput => isRecord(value) && typeof value.text === 'string' && value.text.length <= 65_536 &&
  Array.isArray(value.tokenIds) && value.tokenIds.length > 0 && value.tokenIds.length <= 512 && value.tokenIds.every(token => Number.isInteger(token) && token >= 0 && token < 8192);

/** One intervention the screen can restore: head scaling at all or one position, a one-hot or even attention redirect, or a residual patch with its source. */
function checkSavedIntervention(patch: unknown, version: 1 | 2, source: unknown): void {
  requireInput(isRecord(patch) && typeof patch.id === 'string' && patch.id.length <= 128 && Number.isInteger(patch.layer), 'Invalid saved intervention.');
  requireInput(source === undefined || (version === 2 && patch.kind === 'residual.replace'), 'Only a saved residual patch records a source input.');
  if (patch.kind === 'head.scale') {
    requireInput(Number.isInteger(patch.head) && Number.isFinite(patch.factor) && isRecord(patch.tokens) &&
      (patch.tokens.kind === 'all' || (patch.tokens.kind === 'indices' && Array.isArray(patch.tokens.indices) && patch.tokens.indices.length === 1)), 'Save an all-position or single-position head intervention.');
    return;
  }
  requireInput(version === 2, 'This version supports one saved head-scaling intervention.');
  if (patch.kind === 'attention.override') {
    const weights = patch.weights, query = patch.query;
    requireInput(Number.isInteger(patch.head) && Number.isInteger(query) && Array.isArray(weights) && weights.length <= 512 && weights.every(weight => weight === 0 || weight === 1),
      'Save a one-position or even attention redirect.');
    const ones = (weights as number[]).flatMap((weight, i) => weight === 1 ? [i] : []);
    requireInput(ones.length === 1 || (ones.length === (query as number) + 1 && ones.every((position, i) => position === i)), 'Save a one-position or even attention redirect.');
    return;
  }
  requireInput(patch.kind === 'residual.replace' && residualHooks.includes(patch.stage as never) && Number.isInteger(patch.position) && isVector(patch.values), 'Unsupported saved intervention.');
  requireInput(isSavedInput(source), 'A saved residual patch must record its source input.');
}

function checkDocument(value: unknown): asserts value is ExperimentDocument {
  requireInput(isRecord(value) && value.format === 'transformer-microscope-experiment' && (value.schemaVersion === 1 || value.schemaVersion === 2), 'Unsupported experiment file. Use a version 1 or 2 Transformer Microscope export.');
  requireInput(typeof value.capturedAt === 'string' && value.capturedAt.length <= 40 && Number.isFinite(Date.parse(value.capturedAt)), 'Invalid experiment capture time.');
  const { checkpoint, input, observation, results } = value;
  requireInput(isRecord(checkpoint) && typeof checkpoint.modelId === 'string' && checkpoint.modelId.length > 0 && checkpoint.modelId.length <= 128 &&
    checkpoint.architectureVersion === 'preln-absolute-tied-v1' && ['modelRevision', 'weightsSha256', 'tokenizerSha256'].every(key => typeof checkpoint[key] === 'string' && hex.test(checkpoint[key] as string)), 'Invalid saved checkpoint identity.');
  requireInput(isRecord(input) && typeof input.text === 'string' && input.text.length <= 65_536 && Array.isArray(input.tokenIds) && input.tokenIds.length > 0 && input.tokenIds.length <= 512 &&
    input.tokenIds.every(token => Number.isInteger(token) && token >= 0 && token < 8192), 'Invalid saved input.');
  requireInput(isRecord(observation) && Number.isInteger(observation.predictionPosition) && Number.isInteger(observation.targetTokenId) &&
    Array.isArray(observation.interventions) && observation.interventions.length === 1, 'Experiment files record exactly one intervention.');
  checkSavedIntervention(observation.interventions[0], value.schemaVersion as 1 | 2, observation.source);
  requireInput(isRecord(results) && isVector(results.baselineLogits) && isVector(results.modifiedLogits) && results.baselineLogits.length === results.modifiedLogits.length &&
    results.temperature === 1 && results.logarithmUnit === 'nats' && Number.isInteger(results.topK) && (results.backend === 'cpu' || results.backend === 'webgl'), 'Invalid saved prediction results.');
  const metrics = results.metrics, precision = results.precision;
  requireInput(isRecord(metrics) && ['jsDivergence', 'maxAbsoluteLogitChange', 'targetLogProbabilityDelta'].every(key => Number.isFinite(metrics[key])), 'Invalid saved comparison metrics.');
  requireInput(isRecord(precision) && precision.dtype === 'float32' && ['float32Textures', 'webglPacking'].every(key => precision[key] === null || typeof precision[key] === 'boolean'), 'Invalid saved runtime precision.');
}

export function parseExperiment(text: string): ExperimentDocument {
  requireInput(new TextEncoder().encode(text).byteLength <= MAX_EXPERIMENT_BYTES, 'Experiment files must be at most 1 MiB.');
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error('The experiment file is not valid JSON.'); }
  checkDocument(value);
  return structuredClone(value);
}

export function validateExperiment(document: ExperimentDocument, manifest: ModelManifest, tokenizer: SymbolTokenizer): void {
  checkDocument(document);
  requireInput(['modelId', 'modelRevision', 'architectureVersion', 'weightsSha256', 'tokenizerSha256'].every(key =>
    document.checkpoint[key as keyof ExperimentDocument['checkpoint']] === manifest[key as keyof ExperimentDocument['checkpoint']]),
  'The saved checkpoint does not match the installed model. This experiment cannot be replayed with different weights or symbols.');
  validateTokens(document.input.tokenIds, manifest.config);
  requireInput(JSON.stringify(tokenize(document.input.text, tokenizer, manifest.config.contextLength)) === JSON.stringify(document.input.tokenIds), 'Saved text and token IDs do not agree.');
  const { observation, results } = document;
  validateInterventions(observation.interventions, manifest.config, document.input.tokenIds.length);
  if (observation.source) {
    validateTokens(observation.source.tokenIds, manifest.config);
    requireInput(JSON.stringify(tokenize(observation.source.text, tokenizer, manifest.config.contextLength)) === JSON.stringify(observation.source.tokenIds), 'Saved source text and token IDs do not agree.');
    requireInput(observation.source.tokenIds.length === document.input.tokenIds.length, 'The saved source input must have the same length as the input.');
  }
  requireInput(observation.predictionPosition >= 0 && observation.predictionPosition < document.input.tokenIds.length, 'Saved prediction position is outside the input.');
  const computed = comparePredictions(results.baselineLogits, results.modifiedLogits, tokenizer, results.topK, observation.targetTokenId);
  for (const key of ['jsDivergence', 'maxAbsoluteLogitChange', 'targetLogProbabilityDelta'] as const)
    requireInput(Math.abs(computed.metrics[key]! - results.metrics[key]!) <= 1e-12 * Math.max(1, Math.abs(computed.metrics[key]!)), 'Saved metrics do not agree with the complete logit arrays.');
}

/** observation.source carries token IDs; canonical text is derived like the input's. */
export function createExperiment(manifest: ModelManifest, tokenizer: SymbolTokenizer, tokenIds: number[],
  observation: Omit<ExperimentDocument['observation'], 'source'> & { sourceTokenIds?: number[] },
  baselineLogits: ArrayLike<number>, modifiedLogits: ArrayLike<number>, runtime: { backend: string; precision: RuntimePrecision }): ExperimentDocument {
  const results = comparePredictions(baselineLogits, modifiedLogits, tokenizer, Math.min(8, tokenizer.tokens.length), observation.targetTokenId);
  const document: ExperimentDocument = {
    format: 'transformer-microscope-experiment', schemaVersion: EXPERIMENT_SCHEMA_VERSION, capturedAt: new Date().toISOString(),
    checkpoint: { modelId: manifest.modelId, modelRevision: manifest.modelRevision, architectureVersion: manifest.architectureVersion,
      weightsSha256: manifest.weightsSha256, tokenizerSha256: manifest.tokenizerSha256 },
    input: savedInput(tokenIds, tokenizer),
    observation: { predictionPosition: observation.predictionPosition, targetTokenId: observation.targetTokenId, interventions: structuredClone(observation.interventions),
      ...(observation.sourceTokenIds ? { source: savedInput(observation.sourceTokenIds, tokenizer) } : {}) },
    results: { baselineLogits: Array.from(baselineLogits), modifiedLogits: Array.from(modifiedLogits), metrics: results.metrics, topK: Math.min(8, tokenizer.tokens.length),
      temperature: 1, logarithmUnit: 'nats', backend: runtime.backend as 'cpu' | 'webgl', precision: { ...runtime.precision } },
  };
  validateExperiment(document, manifest, tokenizer);
  return document;
}

const savedInput = (tokenIds: number[], tokenizer: SymbolTokenizer): SavedInput => ({ text: tokenIds.slice(1).map(id => tokenizer.tokens[id]).join(' '), tokenIds: [...tokenIds] });

export function checkReplay(document: ExperimentDocument, baseline: ArrayLike<number>, modified: ArrayLike<number>, currentBackend: string): ReplayReport {
  let matches = true;
  function compare(actual: ArrayLike<number>, expected: number[]): number {
    requireInput(actual.length === expected.length, 'Replay vocabulary mismatch.');
    let maximum = 0;
    for (let i = 0; i < actual.length; i++) {
      requireInput(Number.isFinite(actual[i]), 'Replay contains non-finite logits.');
      const error = Math.abs(actual[i] - expected[i]);
      maximum = Math.max(maximum, error);
      if (error > 2e-5 + 1e-5 * Math.abs(expected[i])) matches = false;
    }
    return maximum;
  }
  return { baselineMaximumError: compare(baseline, document.results.baselineLogits), modifiedMaximumError: compare(modified, document.results.modifiedLogits),
    matches, savedBackend: document.results.backend, currentBackend };
}
