import * as tf from '@tensorflow/tfjs-core';
import { activationKey, EngineError, type ActivationReference, type ModelConfig } from './contracts';
import { TransformerModel, disposeForward } from './model';
import { traceNodeKey, traceRoute, type TraceAddress, type TraceIntervention, type TraceRoute } from './trace-topology';

export interface TraceMetrics { baselineNorm: number; modifiedNorm: number; maxDifference: number; differenceNorm: number; change: 'exact' | 'small' | 'changed' }
export interface TraceMeasurement extends TraceAddress, TraceMetrics { key: string; route: TraceRoute }
export interface TraceReadout extends ActivationReference { baselineLogits: number[]; modifiedLogits: number[]; prediction: boolean }
export interface TraceAttentionRow { baselineWeights: number[]; modifiedWeights: number[]; scores: TraceMetrics }
export interface TraceResidualRow { baselineValues: number[]; copiedValues: number[] }
export interface EffectTraceReport {
  baselineRunId: string; modifiedRunId: string; predictionPosition: number; tokenIds: number[]; config: ModelConfig;
  modelRevision: string; runtimeGeneration: string; backend: string; intervention: TraceIntervention; hasComputationalPath: boolean;
  attentionRow?: TraceAttentionRow;
  residualRow?: TraceResidualRow;
  nodes: TraceMeasurement[]; readouts: TraceReadout[]; elapsedMs: number;
}
export const TRACE_CAPTURE_BUDGET = 8 * 1024 * 1024;
export const TRACE_NODE_BUDGET = 8192;
export const traceReferences = (config: ModelConfig): ActivationReference[] => [
  { stage: 'embedding.output' }, { stage: 'final.normalized' }, { stage: 'logits' },
  ...Array.from({ length: config.layers }, (_, layer) => (['block.input', 'attention.normalized', 'attention.weights', 'attention.head.raw', 'attention.head.modified',
    'attention.projected', 'attention.residual', 'mlp.output', 'block.output'] as const).map(stage => ({ layer, stage }))).flat(),
];
export function traceBudget(config: ModelConfig, length: number, redirect = false) {
  const floats = (2 + 8 * config.layers) * length * config.modelDim + config.layers * config.heads * length * length + length * config.vocabularySize;
  const readouts = (1 + 2 * config.layers) * (config.modelDim + config.vocabularySize);
  return { captureBytes: 2 * (floats + readouts + (redirect ? config.heads * length * length : 0)) * 4, nodes: length * (3 + config.layers * (6 + 3 * config.heads)) };
}
export function measureTrace(baseline: ArrayLike<number>, modified: ArrayLike<number>): TraceMetrics {
  if (baseline.length !== modified.length) throw new EngineError('NUMERICAL_FAILURE', 'Trace vector dimensions do not agree.');
  let baseSquare = 0, modifiedSquare = 0, deltaSquare = 0, maxDifference = 0, small = true;
  for (let i = 0; i < baseline.length; i++) {
    const a = baseline[i], b = modified[i], delta = Math.abs(b - a);
    if (!Number.isFinite(a) || !Number.isFinite(b)) throw new EngineError('NUMERICAL_FAILURE', 'Trace produced non-finite values.');
    baseSquare += a * a; modifiedSquare += b * b; deltaSquare += delta * delta; maxDifference = Math.max(maxDifference, delta);
    if (delta > 2e-5 + 1e-5 * Math.abs(a)) small = false;
  }
  return { baselineNorm: Math.sqrt(baseSquare), modifiedNorm: Math.sqrt(modifiedSquare), maxDifference, differenceNorm: Math.sqrt(deltaSquare),
    change: maxDifference === 0 ? 'exact' : small ? 'small' : 'changed' };
}
export function checkTracePrediction(actual: ArrayLike<number>, recorded: ArrayLike<number>) {
  if (actual.length !== recorded.length || Array.from(actual).some((value, i) => !Number.isFinite(value) || !Number.isFinite(recorded[i]) || Math.abs(value - recorded[i]) > 2e-5 + 1e-5 * Math.abs(recorded[i])))
    throw new EngineError('NUMERICAL_FAILURE', 'The trace does not match the completed prediction within the verified tolerance.');
}
interface TracePass { values: Record<string, Float32Array>; lens: number[][] }
/** One forward pass per run. Captures and readout tensors are released before returning owned CPU arrays. */
export async function captureTracePass(model: TransformerModel, tokens: number[], patches: TraceIntervention[], prediction: number, scoreLayer?: number): Promise<TracePass> {
  const references = traceReferences(model.config);
  if (scoreLayer !== undefined) references.push({ layer: scoreLayer, stage: 'attention.scores' });
  const result = model.forward(tokens, patches, references);
  let read: tf.Tensor3D | undefined;
  try {
    const values = Object.fromEntries(await Promise.all(Object.entries(result.captured).map(async ([key, tensor]) => [key, Float32Array.from(await tensor.data())])));
    const residuals: ActivationReference[] = [{ stage: 'embedding.output' }, ...Array.from({ length: model.config.layers }, (_, layer) =>
      [{ stage: 'attention.residual' as const, layer }, { stage: 'block.output' as const, layer }]).flat()];
    read = tf.tidy(() => {
      const rows = tf.concat(residuals.map(reference => tf.slice(result.captured[activationKey(reference)], [0, prediction, 0], [1, 1, model.config.modelDim])), 1) as tf.Tensor3D;
      return model.readOut(rows);
    });
    const logits = await read.data(), size = model.config.vocabularySize;
    if (!logits.every(Number.isFinite)) throw new EngineError('NUMERICAL_FAILURE', 'Trace readout produced non-finite values.');
    return { values, lens: residuals.map((_, i) => Array.from(logits.slice(i * size, (i + 1) * size))) };
  } finally { read?.dispose(); disposeForward(result); }
}
export function summarizeAttentionRow(length: number, patch: TraceIntervention, baseline: TracePass, modified: TracePass): TraceAttentionRow | undefined {
  if (patch.kind !== 'attention.override') return undefined;
  const offset = (patch.head * length + patch.query) * length;
  const row = (pass: TracePass, stage: 'attention.weights' | 'attention.scores') => pass.values[`${patch.layer}:${stage}`].subarray(offset, offset + length);
  return { baselineWeights: Array.from(row(baseline, 'attention.weights')), modifiedWeights: Array.from(row(modified, 'attention.weights')),
    scores: measureTrace(row(baseline, 'attention.scores'), row(modified, 'attention.scores')) };
}
export function summarizeResidualRow(config: ModelConfig, patch: TraceIntervention, baseline: TracePass, modified: TracePass): TraceResidualRow | undefined {
  if (patch.kind !== 'residual.replace') return undefined;
  const offset = patch.position * config.modelDim, key = activationKey(patch);
  const copied = modified.values[key].subarray(offset, offset + config.modelDim);
  if (copied.length !== patch.values.length || Array.from(copied).some((value, i) => value !== Math.fround(patch.values[i])))
    throw new EngineError('NUMERICAL_FAILURE', 'The trace does not reproduce the copied residual vector.');
  return { baselineValues: Array.from(baseline.values[key].subarray(offset, offset + config.modelDim)), copiedValues: Array.from(copied) };
}
export function summarizeTrace(config: ModelConfig, length: number, prediction: number, patch: TraceIntervention, baseline: TracePass, modified: TracePass): TraceMeasurement[] {
  const nodes: TraceMeasurement[] = [];
  for (const reference of traceReferences(config)) {
    const headStage = ['attention.weights', 'attention.head.raw', 'attention.head.modified'].includes(reference.stage);
    const width = reference.stage === 'attention.weights' ? length : headStage ? config.modelDim / config.heads : reference.stage === 'logits' ? config.vocabularySize : config.modelDim;
    for (let position = 0; position < length; position++) for (let head = 0; head < (headStage ? config.heads : 1); head++) {
      const address: TraceAddress = { ...reference, position, ...(headStage ? { head } : {}) };
      const offset = reference.stage === 'attention.weights' ? (head * length + position) * width : (headStage ? position * config.heads + head : position) * width;
      nodes.push({ ...address, key: traceNodeKey(address), route: traceRoute(address, patch, config, length, prediction),
        ...measureTrace(baseline.values[activationKey(reference)].subarray(offset, offset + width), modified.values[activationKey(reference)].subarray(offset, offset + width)) });
    }
  }
  return nodes;
}
