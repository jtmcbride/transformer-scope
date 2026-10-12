import * as tf from '@tensorflow/tfjs-core';
import { TransformerModel, disposeForward } from './model';
import { comparePredictions, type PredictionComparison } from './comparison';
import { activationKey, canReachPrediction, EngineError, requireInput, stages, validateInterventions, validateTokens,
  type ActivationReference, type HeadScaleIntervention, type ModelArtifact } from './contracts';

interface RunDescriptor { tokenIds: number[]; interventions: HeadScaleIntervention[]; baselineRunId?: string; logits?: Float32Array }
export interface RunSummary { runId: string; tokenIds: number[]; modelRevision: string; backend: string; runtimeGeneration: string }
export interface CompareRequest {
  baselineRunId: string; interventions: HeadScaleIntervention[]; predictionPosition?: number; targetTokenId?: number; topK?: number;
}
export interface ComparisonResult extends PredictionComparison {
  baselineRunId: string; modifiedRunId: string; interventions: HeadScaleIntervention[]; predictionPosition: number;
  execution: { strategy: 'identity' | 'full'; elapsedMs: number; backend: string; runtimeGeneration: string };
  hasComputationalPath: boolean;
}
export interface ActivationSlice {
  shape: number[]; axes: string[]; values: Float32Array;
}

function axesFor(stage: ActivationReference['stage']): string[] {
  if (['attention.scores', 'attention.maskedScores', 'attention.weights'].includes(stage)) return ['batch', 'head', 'queryPosition', 'keyPosition'];
  if (['attention.query', 'attention.key', 'attention.value', 'attention.head.raw', 'attention.head.modified'].includes(stage)) return ['batch', 'position', 'head', 'dimension'];
  return ['batch', 'position', stage === 'logits' ? 'tokenId' : 'dimension'];
}

/** Descriptors survive inspection because intermediate tensor materializations are transient. */
export class RunEngine {
  readonly model: TransformerModel;
  readonly runtimeGeneration = crypto.randomUUID();
  private runs = new Map<string, RunDescriptor>();
  private disposed = false;

  constructor(readonly artifact: ModelArtifact) { this.model = new TransformerModel(artifact.manifest.config, artifact.weights); }

  async runBaseline(tokenIds: number[]): Promise<RunSummary> {
    requireInput(!this.disposed, 'Engine has been disposed.');
    validateTokens(tokenIds, this.model.config);
    const result = this.model.forward(tokenIds);
    let logits: Float32Array;
    try { logits = Float32Array.from(await result.logits.data()); }
    finally { disposeForward(result); }
    const runId = this.store({ tokenIds: [...tokenIds], interventions: [], logits });
    return { runId, tokenIds: [...tokenIds], modelRevision: this.artifact.manifest.modelRevision,
      backend: tf.getBackend(), runtimeGeneration: this.runtimeGeneration };
  }

  private descriptor(runId: string): RunDescriptor {
    if (this.disposed) throw new EngineError('STALE_SESSION', 'This model session has ended. Run the input again.');
    const run = this.runs.get(runId);
    if (!run) throw new EngineError('RELEASED_RUN', 'This run is unavailable. Run the input again.');
    return run;
  }

  private store(run: RunDescriptor): string {
    if (this.runs.size >= 64) throw new EngineError('RESOURCE_EXHAUSTION', 'Release an older run before creating another.');
    const id = `${this.runtimeGeneration}/${crypto.randomUUID()}`;
    this.runs.set(id, structuredClone(run));
    return id;
  }

  async compare(request: CompareRequest): Promise<ComparisonResult> {
    const baseline = this.descriptor(request.baselineRunId);
    requireInput(baseline.interventions.length === 0 && !baseline.baselineRunId, 'Comparison requires an unmodified baseline.');
    validateInterventions(request.interventions, this.model.config, baseline.tokenIds.length);
    const position = request.predictionPosition ?? baseline.tokenIds.length - 1;
    requireInput(Number.isInteger(position) && position >= 0 && position < baseline.tokenIds.length, 'Prediction position is outside the input.');
    const patches = structuredClone(request.interventions);
    const effective = patches.filter(p => p.factor !== 1 && (p.tokens.kind === 'all' || p.tokens.indices.length > 0));
    const start = performance.now();
    const size = this.model.config.vocabularySize;
    const original = baseline.logits!.slice(position * size, (position + 1) * size);
    const changed = effective.length ? await this.prediction({ ...baseline, interventions: patches }, position) : original;
    const comparison = comparePredictions(original, changed, this.artifact.tokenizer, request.topK ?? Math.min(8, size), request.targetTokenId);
    const modifiedRunId = this.store({ tokenIds: baseline.tokenIds, interventions: patches, baselineRunId: request.baselineRunId });
    return { ...comparison, baselineRunId: request.baselineRunId, modifiedRunId, interventions: patches, predictionPosition: position,
      hasComputationalPath: effective.some(patch => canReachPrediction(patch, position, this.model.config, baseline.tokenIds.length)),
      execution: { strategy: effective.length ? 'full' : 'identity', elapsedMs: performance.now() - start,
        backend: tf.getBackend(), runtimeGeneration: this.runtimeGeneration } };
  }

  private async prediction(run: RunDescriptor, position: number): Promise<Float32Array> {
    const result = this.model.forward(run.tokenIds, run.interventions);
    const slice = tf.slice(result.logits, [0, position, 0], [1, 1, this.model.config.vocabularySize]);
    try { return Float32Array.from(await slice.data()); }
    finally { slice.dispose(); disposeForward(result); }
  }

  async inspectActivation(runId: string, reference: ActivationReference, selection: Record<string, number> = {}): Promise<ActivationSlice> {
    const run = this.descriptor(runId);
    requireInput(stages.includes(reference.stage), 'Unknown activation stage.');
    const needsLayer = !['embedding.output', 'final.normalized', 'logits'].includes(reference.stage);
    requireInput(needsLayer ? Number.isInteger(reference.layer) && reference.layer! >= 0 && reference.layer! < this.model.config.layers : reference.layer === undefined,
      'Activation reference has an invalid layer.');
    const axes = axesFor(reference.stage);
    requireInput(Object.keys(selection).every(key => axes.includes(key)), 'Selection contains an unknown tensor axis.');
    const result = this.model.forward(run.tokenIds, run.interventions, [reference]);
    let slice: tf.Tensor | undefined;
    try {
      const tensor = result.captured[activationKey(reference)];
      const begin = tensor.shape.map((size, i) => {
        const index = selection[axes[i]];
        requireInput(index === undefined || (Number.isInteger(index) && index >= 0 && index < size), `Invalid ${axes[i]} selection.`);
        return index ?? 0;
      });
      const sizes = tensor.shape.map((size, i) => selection[axes[i]] === undefined ? size : 1);
      requireInput(sizes.reduce((a, b) => a * b, 1) <= 65_536, 'Request a smaller activation slice.');
      slice = tf.slice(tensor, begin, sizes);
      return { shape: sizes, axes, values: Float32Array.from(await slice.data()) };
    } finally { slice?.dispose(); disposeForward(result); }
  }

  releaseRun(runId: string): void { this.runs.delete(runId); }
  dispose(): void { this.disposed = true; this.runs.clear(); this.model.dispose(); }
}
