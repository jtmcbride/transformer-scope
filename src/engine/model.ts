import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import { activationKey, requireInput, validateInterventions, validateTokens,
  type ActivationReference, type HeadScaleIntervention, type ModelConfig, type TensorRecord } from './contracts';
import { validateWeightRecords } from './weights';

export interface ForwardResult { logits: tf.Tensor3D; captured: Record<string, tf.Tensor> }
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

  forward(tokenIds: readonly number[], patches: readonly HeadScaleIntervention[] = [], capture: readonly ActivationReference[] = []): ForwardResult {
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
      const t = tokenIds.length, d = c.modelDim, h = c.heads, dh = d / h;
      const ids = tf.tensor1d([...tokenIds], 'int32');
      const tokenEmbeddings = tf.gather(this.weights.tokenEmbedding, ids);
      const positionEmbeddings = tf.gather(this.weights.positionEmbedding, tf.range(0, t, 1, 'int32'));
      let x = save('embedding.output', tf.reshape(tf.add(tokenEmbeddings, positionEmbeddings), [1, t, d]) as tf.Tensor3D);
      const maskValues = new Float32Array(t * t);
      for (let query = 0; query < t; query++) for (let key = query + 1; key < t; key++) maskValues[query * t + key] = -1e9;
      const mask = tf.tensor2d(maskValues, [t, t]);
      for (let layer = 0; layer < c.layers; layer++) {
        const prefix = `blocks.${layer}.`;
        const at = <T extends tf.Tensor>(stage: string, value: T): T => save(`${layer}:${stage}`, value);
        at('block.input', x);
        const u = at('attention.normalized', this.norm(x, prefix + 'ln1'));
        const project = (name: string, stage: string) => at(stage, tf.reshape(this.linear(u, prefix + name), [1, t, h, dh]) as tf.Tensor4D);
        const q = project('q', 'attention.query'), k = project('k', 'attention.key'), v = project('v', 'attention.value');
        const scores = at('attention.scores', tf.div(tf.matMul(tf.transpose(q, [0, 2, 1, 3]), tf.transpose(k, [0, 2, 1, 3]), false, true), Math.sqrt(dh)));
        const masked = at('attention.maskedScores', tf.add(scores, mask));
        const a = at('attention.weights', tf.softmax(masked, -1));
        const z = at('attention.head.raw', tf.transpose(tf.matMul(a, tf.transpose(v, [0, 2, 1, 3])), [0, 2, 1, 3]) as tf.Tensor4D);
        const factors = new Float32Array(t * h).fill(1);
        for (const patch of patches) {
          if (patch.layer !== layer) continue;
          const positions = patch.tokens.kind === 'all' ? Array.from({ length: t }, (_, i) => i) : patch.tokens.indices;
          for (const position of positions) factors[position * h + patch.head] = patch.factor;
        }
        const modified = at('attention.head.modified', tf.mul(z, tf.tensor4d(factors, [1, t, h, 1])));
        const projected = at('attention.projected', this.linear(tf.reshape(modified, [1, t, d]) as tf.Tensor3D, prefix + 'output'));
        const residual = at('attention.residual', tf.add(x, projected) as tf.Tensor3D);
        const expanded = this.linear(this.norm(residual, prefix + 'ln2'), prefix + 'mlpIn');
        // Explicit multiplication keeps the cubic derivative off the generic GPU Pow kernel.
        const cubic = tf.mul(expanded, tf.square(expanded));
        const gelu = tf.mul(tf.mul(expanded, .5), tf.add(1, tf.tanh(tf.mul(Math.sqrt(2 / Math.PI),
          tf.add(expanded, tf.mul(.044715, cubic)))))) as tf.Tensor3D;
        const mlp = at('mlp.output', this.linear(gelu, prefix + 'mlpOut'));
        x = at('block.output', tf.add(residual, mlp) as tf.Tensor3D);
      }
      const final = save('final.normalized', this.norm(x, 'finalNorm'));
      const logits = save('logits', tf.reshape(tf.matMul(tf.reshape(final, [t, d]) as tf.Tensor2D,
        this.weights.tokenEmbedding as tf.Tensor2D, false, true), [1, t, c.vocabularySize]) as tf.Tensor3D);
      return { logits, captured };
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
