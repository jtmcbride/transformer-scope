import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import { activationKey, requireInput, validateInterventions, validateTokens,
  type ActivationReference, type HeadScaleIntervention, type Intervention, type ModelConfig, type ResidualHook, type TensorRecord } from './contracts';
import { validateWeightRecords } from './weights';

export interface ForwardResult { logits: tf.Tensor3D; captured: Record<string, tf.Tensor> }
export interface LayerBaseline { rawHeadOutputs: tf.Tensor4D; attentionResidual: tf.Tensor3D }
type Save = <T extends tf.Tensor>(key: string, value: T) => T;
let instance = 0;

export class TransformerModel {
  readonly weights: Record<string, tf.Variable> = {};
  readonly config: ModelConfig;
  private disposed = false;

  constructor(config: ModelConfig, weights: Record<string, TensorRecord>) {
    validateWeightRecords(weights, config);
    this.config = structuredClone(config);
    const prefix = `model-${++instance}/`;
    try {
      for (const [name, record] of Object.entries(weights))
        this.weights[name] = tf.tidy(() => tf.variable(tf.tensor(record.values, record.shape, 'float32'), true, prefix + name));
    } catch (error) { this.dispose(); throw error; }
  }

  private norm(x: tf.Tensor3D, name: string): tf.Tensor3D {
    const centered = tf.sub(x, tf.mean(x, -1, true));
    const variance = tf.mean(tf.square(centered), -1, true);
    return tf.add(tf.mul(tf.div(centered, tf.sqrt(tf.add(variance, this.config.layerNormEpsilon))),
      this.weights[name + '.scale']), this.weights[name + '.bias']) as tf.Tensor3D;
  }

  private linear(x: tf.Tensor3D, name: string): tf.Tensor3D {
    const [b, t, d] = x.shape;
    const outputDim = this.weights[name + '.bias'].shape[0];
    return tf.reshape(tf.add(tf.matMul(tf.reshape(x, [b * t, d]), this.weights[name + '.weight'] as tf.Tensor2D),
      this.weights[name + '.bias']), [b, t, outputDim]) as tf.Tensor3D;
  }

  private mask(length: number): tf.Tensor2D {
    const values = new Float32Array(length * length);
    for (let query = 0; query < length; query++) for (let key = query + 1; key < length; key++) values[query * length + key] = -1e9;
    return tf.tensor2d(values, [length, length]);
  }

  /** Rows selected by residual.replace take the submitted values exactly; every other row is returned unchanged. */
  private replaced(x: tf.Tensor3D, layer: number, stage: ResidualHook, patches: readonly Intervention[]): tf.Tensor3D {
    const selected = patches.filter(p => p.kind === 'residual.replace' && p.layer === layer && p.stage === stage);
    if (!selected.length) return x;
    const [, length, dimension] = x.shape;
    const values = new Float32Array(length * dimension), mask = new Uint8Array(length * dimension);
    for (const patch of selected) if (patch.kind === 'residual.replace') {
      values.set(patch.values, patch.position * dimension);
      mask.fill(1, patch.position * dimension, (patch.position + 1) * dimension);
    }
    return tf.where(tf.tensor3d(mask, [1, length, dimension], 'bool'), tf.tensor3d(values, [1, length, dimension]), x) as tf.Tensor3D;
  }

  private scaledHeads(raw: tf.Tensor4D, patches: readonly HeadScaleIntervention[]): tf.Tensor4D {
    const [, length, heads] = raw.shape;
    const factors = new Float32Array(length * heads).fill(1);
    for (const patch of patches) {
      const positions = patch.tokens.kind === 'all' ? Array.from({ length }, (_, i) => i) : patch.tokens.indices;
      for (const position of positions) factors[position * heads + patch.head] = patch.factor;
    }
    return tf.mul(raw, tf.tensor4d(factors, [1, length, heads, 1])) as tf.Tensor4D;
  }

  private finishBlock(residual: tf.Tensor3D, layer: number, save: Save): tf.Tensor3D {
    const prefix = `blocks.${layer}.`;
    const expanded = this.linear(this.norm(residual, prefix + 'ln2'), prefix + 'mlpIn');
    const cubic = tf.mul(expanded, tf.square(expanded));
    const gelu = tf.mul(tf.mul(expanded, .5), tf.add(1, tf.tanh(tf.mul(Math.sqrt(2 / Math.PI),
      tf.add(expanded, tf.mul(.044715, cubic)))))) as tf.Tensor3D;
    const mlp = save(`${layer}:mlp.output`, this.linear(gelu, prefix + 'mlpOut'));
    return save(`${layer}:block.output`, tf.add(residual, mlp) as tf.Tensor3D);
  }

  private block(input: tf.Tensor3D, layer: number, patches: readonly Intervention[], mask: tf.Tensor2D, save: Save): tf.Tensor3D {
    const c = this.config;
    const [, t, d] = input.shape, h = c.heads, dh = d / h;
    const prefix = `blocks.${layer}.`;
    const at = <T extends tf.Tensor>(stage: string, value: T): T => save(`${layer}:${stage}`, value);
    const x = at('block.input', this.replaced(input, layer, 'block.input', patches));
    const u = at('attention.normalized', this.norm(x, prefix + 'ln1'));
    const project = (name: string, stage: string) => at(stage, tf.reshape(this.linear(u, prefix + name), [1, t, h, dh]) as tf.Tensor4D);
    const q = project('q', 'attention.query'), k = project('k', 'attention.key'), v = project('v', 'attention.value');
    const scores = at('attention.scores', tf.div(tf.matMul(tf.transpose(q, [0, 2, 1, 3]), tf.transpose(k, [0, 2, 1, 3]), false, true), Math.sqrt(dh)));
    const masked = at('attention.maskedScores', tf.add(scores, mask));
    const a = at('attention.weights', tf.softmax(masked, -1));
    const raw = at('attention.head.raw', tf.transpose(tf.matMul(a, tf.transpose(v, [0, 2, 1, 3])), [0, 2, 1, 3]) as tf.Tensor4D);
    const modified = at('attention.head.modified', this.scaledHeads(raw, patches.filter((p): p is HeadScaleIntervention => p.kind === 'head.scale' && p.layer === layer)));
    const projected = at('attention.projected', this.linear(tf.reshape(modified, [1, t, d]) as tf.Tensor3D, prefix + 'output'));
    const residual = at('attention.residual', this.replaced(tf.add(x, projected) as tf.Tensor3D, layer, 'attention.residual', patches));
    return this.finishBlock(residual, layer, save);
  }

  private logits(hidden: tf.Tensor3D, save: Save): tf.Tensor3D {
    const final = save('final.normalized', this.norm(hidden, 'finalNorm'));
    return save('logits', this.unembed(final));
  }

  private unembed(normalized: tf.Tensor3D): tf.Tensor3D {
    const [, length, dimension] = normalized.shape;
    return tf.reshape(tf.matMul(tf.reshape(normalized, [length, dimension]) as tf.Tensor2D,
      this.weights.tokenEmbedding as tf.Tensor2D, false, true), [1, length, this.config.vocabularySize]) as tf.Tensor3D;
  }

  /** The token and position embedding rows summed at one input position, and the token row's cosine similarity to every token embedding. */
  embeddingParts(tokenId: number, position: number): { token: tf.Tensor1D; position: tf.Tensor1D; sum: tf.Tensor1D; similarity: tf.Tensor1D } {
    requireInput(!this.disposed, 'Model has been disposed.');
    return tf.tidy(() => {
      const table = this.weights.tokenEmbedding as tf.Tensor2D;
      const token = tf.reshape(tf.gather(table, [tokenId]), [-1]) as tf.Tensor1D;
      const place = tf.reshape(tf.gather(this.weights.positionEmbedding, [position]), [-1]) as tf.Tensor1D;
      const unit = tf.div(table, tf.norm(table, 'euclidean', -1, true));
      const similarity = tf.reshape(tf.matMul(unit, tf.reshape(tf.div(token, tf.norm(token)), [-1, 1])), [-1]) as tf.Tensor1D;
      return { token, position: place, sum: tf.add(token, place) as tf.Tensor1D, similarity };
    });
  }

  /** Final normalization and tied unembedding applied to arbitrary residual rows [1, rows, d]: the logit lens. */
  readOut(residuals: tf.Tensor3D): tf.Tensor3D {
    requireInput(!this.disposed, 'Model has been disposed.');
    return tf.tidy(() => this.unembed(this.norm(residuals, 'finalNorm')));
  }

  private validateCached(cache: LayerBaseline, layer: number, patches: readonly HeadScaleIntervention[]): void {
    requireInput(!this.disposed, 'Model has been disposed.');
    const [, length] = cache.rawHeadOutputs.shape;
    const c = this.config;
    requireInput(Number.isInteger(layer) && layer >= 0 && layer < c.layers, 'Invalid cached layer.');
    requireInput(length > 0 && length <= c.contextLength &&
      JSON.stringify(cache.rawHeadOutputs.shape) === JSON.stringify([1, length, c.heads, c.modelDim / c.heads]) &&
      JSON.stringify(cache.attentionResidual.shape) === JSON.stringify([1, length, c.modelDim]), 'Invalid cached tensor shapes.');
    validateInterventions(patches, c, length);
    requireInput(patches.every(patch => patch.kind === 'head.scale' && patch.layer === layer), 'A cached residual accepts head scaling in one layer only.');
  }

  private scaledResidual(cache: LayerBaseline, layer: number, patches: readonly HeadScaleIntervention[]): tf.Tensor3D {
    const [, length, , headDimension] = cache.rawHeadOutputs.shape;
    const dimension = this.config.modelDim;
    // Always return an owned tensor, including the factor-1 case.
    let residual = tf.clone(cache.attentionResidual);
    for (const patch of patches) {
      if (patch.factor === 1 || (patch.tokens.kind === 'indices' && patch.tokens.indices.length === 0)) continue;
      const head = tf.reshape(tf.slice(cache.rawHeadOutputs, [0, 0, patch.head, 0], [1, length, 1, headDimension]), [length, headDimension]);
      let delta = tf.mul(head, patch.factor - 1);
      if (patch.tokens.kind === 'indices') {
        const values = new Float32Array(length);
        for (const position of patch.tokens.indices) values[position] = 1;
        delta = tf.mul(delta, tf.tensor2d(values, [length, 1]));
      }
      const projection = tf.slice(this.weights[`blocks.${layer}.output.weight`], [patch.head * headDimension, 0], [headDimension, dimension]) as tf.Tensor2D;
      const projected = tf.reshape(tf.matMul(delta as tf.Tensor2D, projection), [1, length, dimension]);
      residual = tf.add(residual, projected) as tf.Tensor3D;
    }
    return residual;
  }

  /** Caller owns the returned tensor; borrowed cache tensors are never returned. */
  inspectCached(cache: LayerBaseline, layer: number, patches: readonly HeadScaleIntervention[], stage: ActivationReference['stage']): tf.Tensor | undefined {
    this.validateCached(cache, layer, patches);
    if (!['attention.head.raw', 'attention.head.modified', 'attention.projected', 'attention.residual'].includes(stage)) return undefined;
    return tf.tidy(() => {
      if (stage === 'attention.head.raw') return tf.clone(cache.rawHeadOutputs);
      if (stage === 'attention.head.modified') return this.scaledHeads(cache.rawHeadOutputs, patches);
      if (stage === 'attention.residual') return this.scaledResidual(cache, layer, patches);
      const [, length] = cache.rawHeadOutputs.shape;
      return this.linear(tf.reshape(this.scaledHeads(cache.rawHeadOutputs, patches), [1, length, this.config.modelDim]) as tf.Tensor3D, `blocks.${layer}.output`);
    });
  }

  /** Finish the edited block's MLP, then every subsequent block and unembedding. */
  forwardCached(cache: LayerBaseline, layer: number, patches: readonly HeadScaleIntervention[], capture: readonly ActivationReference[] = []): ForwardResult {
    this.validateCached(cache, layer, patches);
    const requested = new Set(capture.map(activationKey));
    return tf.tidy(() => {
      const captured: Record<string, tf.Tensor> = {};
      const save: Save = (key, value) => { if (requested.has(key)) captured[key] = value; return value; };
      const residual = save(`${layer}:attention.residual`, this.scaledResidual(cache, layer, patches));
      let hidden = this.finishBlock(residual, layer, save);
      const mask = this.mask(cache.rawHeadOutputs.shape[1]);
      for (let next = layer + 1; next < this.config.layers; next++) hidden = this.block(hidden, next, [], mask, save);
      return { logits: this.logits(hidden, save), captured };
    });
  }

  forward(tokenIds: readonly number[], patches: readonly Intervention[] = [], capture: readonly ActivationReference[] = []): ForwardResult {
    requireInput(!this.disposed, 'Model has been disposed.');
    validateTokens(tokenIds, this.config);
    validateInterventions(patches, this.config, tokenIds.length);
    const requested = new Set(capture.map(activationKey));
    return tf.tidy(() => {
      const captured: Record<string, tf.Tensor> = {};
      const save = <T extends tf.Tensor>(key: string, value: T): T => {
        if (requested.has(key)) captured[key] = value;
        return value;
      };
      const c = this.config;
      const t = tokenIds.length, d = c.modelDim;
      const ids = tf.tensor1d([...tokenIds], 'int32');
      const tokenEmbeddings = tf.gather(this.weights.tokenEmbedding, ids);
      const positionEmbeddings = tf.gather(this.weights.positionEmbedding, tf.range(0, t, 1, 'int32'));
      let x = save('embedding.output', tf.reshape(tf.add(tokenEmbeddings, positionEmbeddings), [1, t, d]) as tf.Tensor3D);
      const mask = this.mask(t);
      for (let layer = 0; layer < c.layers; layer++) x = this.block(x, layer, patches, mask, save);
      return { logits: this.logits(x, save), captured };
    });
  }

  loss(tokenIds: readonly number[], targets: readonly number[]): tf.Scalar {
    validateTokens(targets, this.config);
    requireInput(targets.length === tokenIds.length, 'Targets must match the input sequence length.');
    return tf.tidy(() => {
      const { logits } = this.forward(tokenIds);
      const labels = tf.oneHot(tf.tensor1d([...targets], 'int32'), this.config.vocabularySize);
      return tf.neg(tf.mean(tf.sum(tf.mul(labels, tf.logSoftmax(tf.reshape(logits, [targets.length, this.config.vocabularySize]))), -1))) as tf.Scalar;
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    Object.values(this.weights).forEach(weight => weight.dispose());
  }
}

export function disposeForward(result: ForwardResult): void {
  new Set([result.logits, ...Object.values(result.captured)]).forEach(tensor => tensor.dispose());
}
