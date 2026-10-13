import * as tf from '@tensorflow/tfjs-core';
import { TransformerModel, disposeForward, type ForwardResult } from './model';
import { BaselineCache, type CacheLease, type CacheStats } from './cache';
import { effectiveInterventions, planExecution, type ExecutionPreference } from './execution';
import { comparePredictions, type PredictionComparison } from './comparison';
import { captureTracePass, checkTracePrediction, summarizeAttentionRow, summarizeResidualRow, summarizeTrace, traceBudget, TRACE_CAPTURE_BUDGET, TRACE_NODE_BUDGET, type EffectTraceReport } from './trace';
import type { TraceIntervention } from './trace-topology';
import { activationKey, canReachPrediction, EngineError, requireInput, stages, validateInterventions, validateTokens,
  type ActivationReference, type HeadScaleIntervention, type Intervention, type ModelArtifact } from './contracts';

interface RunDescriptor { tokenIds: number[]; interventions: Intervention[]; baselineRunId?: string; logits?: Float32Array; executionPreference?: ExecutionPreference;
  observedPrediction?: { position: number; logits: Float32Array } }
export interface RunSummary { runId: string; tokenIds: number[]; modelRevision: string; backend: string; runtimeGeneration: string }
export interface CompareRequest {
  baselineRunId: string; interventions: Intervention[]; predictionPosition?: number; targetTokenId?: number; topK?: number;
  executionPreference?: ExecutionPreference;
}
export interface ComparisonResult extends PredictionComparison {
  baselineRunId: string; modifiedRunId: string; interventions: Intervention[]; predictionPosition: number;
  execution: { strategy: 'identity' | 'full' | 'cached-suffix'; recomputedFromLayer: number | null; elapsedMs: number; backend: string; runtimeGeneration: string };
  hasComputationalPath: boolean;
}
export interface ActivationSlice {
  shape: number[]; axes: string[]; values: Float32Array;
}
/** One all-position head ablation (factor 0) per layer and head, for one prediction. Each entry comes from the comparison path. */
export interface HeadAblationScan {
  predictionPosition: number; targetTokenId: number; baselineLogits: Float32Array;
  entries: { layer: number; head: number; logits: Float32Array; metrics: ComparisonResult['metrics']; topTokenId: number; targetProbability: number; strategy: ComparisonResult['execution']['strategy'] }[];
}
/** Embedding rows are weights: identical for a baseline and every modified run of the same input. */
export interface EmbeddingSlice { position: number; tokenId: number; token: Float32Array; positionEmbedding: Float32Array; sum: Float32Array; similarity: Float32Array }
/** Residual rows at one position read out through the final normalization and unembedding. Only the last row is a model prediction. */
export interface ResidualLens {
  position: number; stages: { stage: 'embedding.output' | 'attention.residual' | 'block.output'; layer: number | null }[];
  norms: Float32Array; logits: Float32Array;
}

function axesFor(stage: ActivationReference['stage']): string[] {
  if (['attention.scores', 'attention.maskedScores', 'attention.weights'].includes(stage)) return ['batch', 'head', 'queryPosition', 'keyPosition'];
  if (['attention.query', 'attention.key', 'attention.value', 'attention.head.raw', 'attention.head.modified'].includes(stage)) return ['batch', 'position', 'head', 'dimension'];
  return ['batch', 'position', stage === 'logits' ? 'tokenId' : 'dimension'];
}

/** Immutable descriptors survive eviction; the cache owns only baseline tensor materializations. */
export class RunEngine {
  private readonly model: TransformerModel;
  private readonly artifact: Pick<ModelArtifact, 'manifest' | 'tokenizer'>;
  private readonly cache: BaselineCache;
  private readonly backend = tf.backend();
  private readonly backendSignature = this.currentBackendSignature();
  readonly runtimeGeneration = crypto.randomUUID();
  private runs = new Map<string, RunDescriptor>();
  private disposed = false;
  private activeOperations = 0;

  constructor(artifact: ModelArtifact, options: { cacheBudgetBytes?: number } = {}) {
    this.artifact = { manifest: structuredClone(artifact.manifest), tokenizer: structuredClone(artifact.tokenizer) };
    this.cache = new BaselineCache(options.cacheBudgetBytes);
    this.model = new TransformerModel(artifact.manifest.config, artifact.weights);
  }

  private currentBackendSignature(): string {
    return JSON.stringify([tf.getBackend(), 'float32', ...(tf.getBackend() === 'webgl' ?
      [tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED'), tf.env().getBool('WEBGL_PACK')] : [])]);
  }

  private assertSession(): void {
    if (this.disposed) throw new EngineError('STALE_SESSION', 'This model session has ended. Run the input again.');
    if (tf.backend() !== this.backend || this.currentBackendSignature() !== this.backendSignature) {
      this.dispose();
      throw new EngineError('STALE_SESSION', 'The computation backend changed. Reload the model and run the input again.');
    }
  }

  private async operation<T>(work: () => Promise<T>): Promise<T> {
    this.assertSession(); this.activeOperations++;
    try { const result = await work(); this.assertSession(); return result; }
    finally { this.activeOperations--; if (this.disposed && this.activeOperations === 0) this.model.dispose(); }
  }

  private cacheIdentity(tokens: readonly number[]): string {
    const m = this.artifact.manifest;
    return JSON.stringify([m.modelRevision, m.architectureVersion, this.runtimeGeneration, this.backendSignature, 'eval', 'absolute-zero-causal', tokens]);
  }

  cacheStats(): CacheStats { return this.cache.stats(); }

  async runBaseline(tokenIds: number[]): Promise<RunSummary> {
    return this.operation(async () => {
      validateTokens(tokenIds, this.model.config); this.requireCapacity();
      const tokens = [...tokenIds];
      const capture: ActivationReference[] = this.cache.budgetBytes ? Array.from({ length: this.model.config.layers }, (_, layer) =>
        [{ stage: 'attention.head.raw' as const, layer }, { stage: 'attention.residual' as const, layer }]).flat() : [];
      const result = this.model.forward(tokens, [], capture);
      try {
        const logits = Float32Array.from(await result.logits.data());
        if (!logits.every(Number.isFinite)) throw new EngineError('NUMERICAL_FAILURE', 'The baseline produced non-finite logits. Check the model and backend.');
        const runId = this.store({ tokenIds: tokens, interventions: [], logits });
        try {
          if (capture.length) {
            const layers = tf.tidy(() => Array.from({ length: this.model.config.layers }, (_, layer) => ({
              rawHeadOutputs: tf.clone(result.captured[`${layer}:attention.head.raw`]) as tf.Tensor4D,
              attentionResidual: tf.clone(result.captured[`${layer}:attention.residual`]) as tf.Tensor3D,
            })));
            this.cache.put(runId, this.cacheIdentity(tokens), layers);
          }
          return { runId, tokenIds: [...tokens], modelRevision: this.artifact.manifest.modelRevision,
            backend: tf.getBackend(), runtimeGeneration: this.runtimeGeneration };
        } catch (error) { this.releaseRun(runId); throw error; }
      } finally { disposeForward(result); }
    });
  }

  private descriptor(runId: string): RunDescriptor {
    this.assertSession();
    requireInput(typeof runId === 'string', 'Run ID must be a string.');
    if (!runId.startsWith(`${this.runtimeGeneration}/`)) throw new EngineError('STALE_SESSION', 'This run belongs to another model session. Run the input again.');
    const run = this.runs.get(runId);
    if (!run) throw new EngineError('RELEASED_RUN', 'This run is unavailable. Run the input again.');
    return run;
  }

  private store(run: RunDescriptor): string {
    this.assertSession(); this.requireCapacity();
    const id = `${this.runtimeGeneration}/${crypto.randomUUID()}`;
    this.runs.set(id, structuredClone(run));
    return id;
  }

  private requireCapacity(): void {
    if (this.runs.size >= 64) throw new EngineError('RESOURCE_EXHAUSTION', 'Release an older run before creating another.');
  }

  async compare(request: CompareRequest): Promise<ComparisonResult> {
    return this.operation(async () => {
      const baselineRunId = request.baselineRunId;
      const baseline = this.descriptor(baselineRunId);
      this.requireCapacity();
      requireInput(baseline.interventions.length === 0 && !baseline.baselineRunId, 'Comparison requires an unmodified baseline.');
      validateInterventions(request.interventions, this.model.config, baseline.tokenIds.length);
      const position = request.predictionPosition ?? baseline.tokenIds.length - 1;
      requireInput(Number.isInteger(position) && position >= 0 && position < baseline.tokenIds.length, 'Prediction position is outside the input.');
      const size = this.model.config.vocabularySize;
      requireInput(request.topK === undefined || (Number.isInteger(request.topK) && request.topK > 0 && request.topK <= size), 'Invalid prediction display limit.');
      requireInput(request.targetTokenId === undefined || (Number.isInteger(request.targetTokenId) && request.targetTokenId >= 0 && request.targetTokenId < size), 'Target token is outside the vocabulary.');
      requireInput(request.executionPreference === undefined || request.executionPreference === 'auto' || request.executionPreference === 'full', 'Invalid execution preference.');
      const patches = structuredClone(request.interventions);
      const effective = effectiveInterventions(patches);
      const preference = request.executionPreference;
      const targetTokenId = request.targetTokenId;
      const topK = request.topK ?? Math.min(8, size);
      const start = performance.now();
      const original = baseline.logits!.slice(position * size, (position + 1) * size);
      let plan = planExecution(patches, true, preference);
      let lease: CacheLease | undefined;
      if (plan.strategy === 'cached-suffix') {
        lease = this.cache.acquire(baselineRunId, this.cacheIdentity(baseline.tokenIds));
        if (!lease) plan = planExecution(patches, false, preference);
      }
      try {
        let changed: Float32Array = original;
        if (plan.strategy !== 'identity') {
          const result = plan.strategy === 'cached-suffix'
            ? this.model.forwardCached(lease!.layers[plan.fromLayer], plan.fromLayer, plan.patches)
            : this.model.forward(baseline.tokenIds, patches);
          changed = await this.prediction(result, position);
        }
        const comparison = comparePredictions(original, changed, this.artifact.tokenizer, topK, targetTokenId);
        const modifiedRunId = this.store({ tokenIds: baseline.tokenIds, interventions: patches, baselineRunId, executionPreference: preference,
          observedPrediction: { position, logits: changed } });
        return { ...comparison, baselineRunId, modifiedRunId, interventions: patches, predictionPosition: position,
          hasComputationalPath: effective.some(patch => canReachPrediction(patch, position, this.model.config, baseline.tokenIds.length)),
          execution: { strategy: plan.strategy, recomputedFromLayer: plan.fromLayer, elapsedMs: performance.now() - start,
            backend: tf.getBackend(), runtimeGeneration: this.runtimeGeneration } };
      } finally { lease?.release(); }
    });
  }

  private async prediction(result: ForwardResult, position: number): Promise<Float32Array> {
    let slice: tf.Tensor | undefined;
    try {
      slice = tf.slice(result.logits, [0, position, 0], [1, 1, this.model.config.vocabularySize]);
      const values = Float32Array.from(await slice.data());
      if (!values.every(Number.isFinite)) throw new EngineError('NUMERICAL_FAILURE', 'The intervention produced non-finite logits. Check the model and backend.');
      return values;
    } finally { slice?.dispose(); disposeForward(result); }
  }

  async inspectActivation(runId: string, reference: ActivationReference, selection: Record<string, number> = {}): Promise<ActivationSlice> {
    return this.operation(async () => {
      const run = this.descriptor(runId);
      reference = structuredClone(reference); selection = { ...selection };
      requireInput(stages.includes(reference.stage), 'Unknown activation stage.');
      const needsLayer = !['embedding.output', 'final.normalized', 'logits'].includes(reference.stage);
      requireInput(needsLayer ? Number.isInteger(reference.layer) && reference.layer! >= 0 && reference.layer! < this.model.config.layers : reference.layer === undefined,
        'Activation reference has an invalid layer.');
      const axes = axesFor(reference.stage);
      requireInput(Object.keys(selection).every(key => axes.includes(key)), 'Selection contains an unknown tensor axis.');
      const source = run.baselineRunId ?? runId;
      const effective = effectiveInterventions(run.interventions);
      // Cached baseline tensors describe unreplaced residuals, so residual replacement always recomputes.
      const heads = effective.filter((p): p is HeadScaleIntervention => p.kind === 'head.scale');
      const lease = run.executionPreference === 'full' || heads.length !== effective.length ? undefined : this.cache.acquire(source, this.cacheIdentity(run.tokenIds));
      const firstAffected = effective.reduce((first, patch) => Math.min(first, patch.layer), Infinity);
      let result: ForwardResult | undefined;
      let owned: tf.Tensor | undefined;
      let slice: tf.Tensor | undefined;
      try {
        if (lease && reference.layer !== undefined && reference.layer <= firstAffected) {
          owned = this.model.inspectCached(lease.layers[reference.layer], reference.layer,
            heads.filter(p => p.layer === reference.layer), reference.stage);
        }
        if (!owned) {
          const plan = planExecution(run.interventions, lease !== undefined);
          const fromSuffix = plan.strategy === 'cached-suffix' && ((reference.layer === undefined && ['final.normalized', 'logits'].includes(reference.stage)) ||
            (reference.layer !== undefined && reference.layer > plan.fromLayer) ||
            (reference.layer === plan.fromLayer && ['mlp.output', 'block.output'].includes(reference.stage)));
          result = fromSuffix && plan.strategy === 'cached-suffix'
            ? this.model.forwardCached(lease!.layers[plan.fromLayer], plan.fromLayer, plan.patches, [reference])
            : this.model.forward(run.tokenIds, run.interventions, [reference]);
        }
        const tensor = owned ?? result!.captured[activationKey(reference)];
        const begin = tensor.shape.map((size, i) => {
          const index = selection[axes[i]];
          requireInput(index === undefined || (Number.isInteger(index) && index >= 0 && index < size), `Invalid ${axes[i]} selection.`);
          return index ?? 0;
        });
        const sizes = tensor.shape.map((size, i) => selection[axes[i]] === undefined ? size : 1);
        requireInput(sizes.reduce((a, b) => a * b, 1) <= 65_536, 'Request a smaller activation slice.');
        slice = tf.slice(tensor, begin, sizes);
        return { shape: sizes, axes, values: Float32Array.from(await slice.data()) };
      } finally { slice?.dispose(); owned?.dispose(); if (result) disposeForward(result); lease?.release(); }
    });
  }

  async inspectEffectTrace(baselineRunId: string, modifiedRunId: string, predictionPosition: number): Promise<EffectTraceReport> {
    return this.operation(async () => {
      const baseline = this.descriptor(baselineRunId), modified = this.descriptor(modifiedRunId);
      requireInput(!baseline.baselineRunId && !baseline.interventions.length && modified.baselineRunId === baselineRunId &&
        JSON.stringify(baseline.tokenIds) === JSON.stringify(modified.tokenIds), 'Trace requires a comparison and its own baseline.');
      requireInput(modified.interventions.length === 1, 'Tracing supports one intervention.');
      requireInput(Number.isInteger(predictionPosition) && predictionPosition === modified.observedPrediction?.position, 'Trace position must match the completed comparison.');
      const patch = modified.interventions[0] as TraceIntervention, config = this.model.config, length = baseline.tokenIds.length;
      const scoreLayer = patch.kind === 'attention.override' ? patch.layer : undefined;
      const budget = traceBudget(config, length, scoreLayer !== undefined);
      if (budget.captureBytes > TRACE_CAPTURE_BUDGET || budget.nodes > TRACE_NODE_BUDGET) throw new EngineError('RESOURCE_EXHAUSTION', 'This model/input exceeds the trace measurement budget.');
      const start = performance.now();
      const original = await captureTracePass(this.model, baseline.tokenIds, [], predictionPosition, scoreLayer);
      const changed = await captureTracePass(this.model, modified.tokenIds, [patch], predictionPosition, scoreLayer);
      const size = config.vocabularySize, offset = predictionPosition * size;
      const baseLogits = baseline.logits!.slice(offset, offset + size), changedLogits = modified.observedPrediction!.logits;
      checkTracePrediction(original.values.logits.subarray(offset, offset + size), baseLogits);
      checkTracePrediction(changed.values.logits.subarray(offset, offset + size), changedLogits);
      const residuals: ActivationReference[] = [{ stage: 'embedding.output' }, ...Array.from({ length: config.layers }, (_, layer) =>
        [{ stage: 'attention.residual' as const, layer }, { stage: 'block.output' as const, layer }]).flat()];
      return { baselineRunId, modifiedRunId, predictionPosition, tokenIds: [...baseline.tokenIds], config: { ...config },
        modelRevision: this.artifact.manifest.modelRevision, runtimeGeneration: this.runtimeGeneration, backend: tf.getBackend(), intervention: structuredClone(patch),
        hasComputationalPath: canReachPrediction(patch, predictionPosition, config, length),
        nodes: summarizeTrace(config, length, predictionPosition, patch, original, changed),
        attentionRow: summarizeAttentionRow(length, patch, original, changed),
        residualRow: summarizeResidualRow(config, patch, original, changed),
        readouts: [...residuals.map((reference, i) => ({ ...reference, prediction: false, baselineLogits: original.lens[i], modifiedLogits: changed.lens[i] })),
          { stage: 'logits', prediction: true, baselineLogits: Array.from(baseLogits), modifiedLogits: Array.from(changedLogits) }], elapsedMs: performance.now() - start };
    });
  }

  /** Ablates every head in turn for one baseline prediction; modified runs are released before returning. */
  async scanHeadAblations(baselineRunId: string, predictionPosition: number, targetTokenId: number): Promise<HeadAblationScan> {
    const { layers, heads } = this.model.config;
    const baselineLogits = await this.inspectActivation(baselineRunId, { stage: 'logits' }, { position: predictionPosition });
    const entries: HeadAblationScan['entries'] = [];
    for (let layer = 0; layer < layers; layer++) for (let head = 0; head < heads; head++) {
      const result = await this.compare({ baselineRunId, predictionPosition, targetTokenId, topK: 1,
        interventions: [{ id: `ablate-${layer}-${head}`, kind: 'head.scale', layer, head, factor: 0, tokens: { kind: 'all' } }] });
      try {
        const logits = await this.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position: predictionPosition });
        entries.push({ layer, head, logits: logits.values, metrics: result.metrics, topTokenId: result.modifiedTopTokens[0].tokenId,
          targetProbability: result.displayTokens.find(token => token.tokenId === targetTokenId)!.probability, strategy: result.execution.strategy });
      } finally { this.releaseRun(result.modifiedRunId); }
    }
    return { predictionPosition, targetTokenId, baselineLogits: baselineLogits.values, entries };
  }

  async inspectEmbedding(runId: string, position: number): Promise<EmbeddingSlice> {
    return this.operation(async () => {
      const run = this.descriptor(runId);
      requireInput(Number.isInteger(position) && position >= 0 && position < run.tokenIds.length, 'Embedding position is outside the input.');
      const parts = this.model.embeddingParts(run.tokenIds[position], position);
      try {
        const [token, positionEmbedding, sum, similarity] = await Promise.all([parts.token.data(), parts.position.data(), parts.sum.data(), parts.similarity.data()]);
        return { position, tokenId: run.tokenIds[position], token: Float32Array.from(token), positionEmbedding: Float32Array.from(positionEmbedding),
          sum: Float32Array.from(sum), similarity: Float32Array.from(similarity) };
      } finally { Object.values(parts).forEach(tensor => tensor.dispose()); }
    });
  }

  async inspectResidualLens(runId: string, position: number): Promise<ResidualLens> {
    return this.operation(async () => {
      const run = this.descriptor(runId);
      requireInput(Number.isInteger(position) && position >= 0 && position < run.tokenIds.length, 'Lens position is outside the input.');
      const stagesRead: ResidualLens['stages'] = [{ stage: 'embedding.output', layer: null }];
      for (let layer = 0; layer < this.model.config.layers; layer++) stagesRead.push({ stage: 'attention.residual', layer }, { stage: 'block.output', layer });
      const references = stagesRead.map(item => item.layer === null ? { stage: item.stage } : { stage: item.stage, layer: item.layer }) as ActivationReference[];
      // A full forward pass keeps this read-out independent of cache state.
      const result = this.model.forward(run.tokenIds, run.interventions, references);
      let rows: tf.Tensor3D | undefined, read: tf.Tensor3D | undefined, norms: tf.Tensor | undefined;
      try {
        rows = tf.tidy(() => tf.concat(references.map(reference => tf.slice(result.captured[activationKey(reference)], [0, position, 0], [1, 1, this.model.config.modelDim])), 1) as tf.Tensor3D);
        read = this.model.readOut(rows);
        norms = tf.tidy(() => tf.sqrt(tf.sum(tf.square(rows!), -1)));
        const [logits, magnitude] = await Promise.all([read.data(), norms.data()]);
        if (!logits.every(Number.isFinite) || !magnitude.every(Number.isFinite)) throw new EngineError('NUMERICAL_FAILURE', 'The residual read-out produced non-finite values.');
        return { position, stages: stagesRead, norms: Float32Array.from(magnitude), logits: Float32Array.from(logits) };
      } finally { rows?.dispose(); read?.dispose(); norms?.dispose(); disposeForward(result); }
    });
  }

  releaseRun(runId: string): void { this.runs.delete(runId); this.cache.delete(runId); }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true; this.runs.clear(); this.cache.clear();
    if (this.activeOperations === 0) this.model.dispose();
  }
}
