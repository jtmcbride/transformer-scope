import { beforeAll, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import reference from '../fixtures/reference.json';
import { TransformerModel, disposeForward } from '../../src/engine/model';
import { type ActivationReference, type HeadScaleIntervention, type TensorRecord, canReachPrediction, tokenize } from '../../src/engine/contracts';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import { comparePredictions } from '../../src/engine/comparison';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });

function assertClose(actual: ArrayLike<number>, expected: ArrayLike<number>, atol = 2e-5, rtol = 1e-5): number {
  expect(actual.length).toBe(expected.length);
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || error > atol + rtol * Math.abs(expected[i]))
      throw new Error(`Value ${i}: actual=${actual[i]}, expected=${expected[i]}, error=${error}`);
    maximum = Math.max(maximum, error);
  }
  return maximum;
}
function references(keys: string[]): ActivationReference[] {
  return keys.map(key => {
    const parts = key.split(':');
    return parts.length === 1 ? { stage: parts[0] as ActivationReference['stage'] } :
      { stage: parts[1] as ActivationReference['stage'], layer: Number(parts[0]) };
  });
}

describe('independent PyTorch parity', () => {
  it('reproduces the deterministic two-head projection example and bias cancellation', () => {
    const result = tf.tidy(() => {
      const projection = tf.tensor2d([[1, 3], [2, 1]]);
      const bias = tf.tensor1d([.5, -.5]);
      const baseline = tf.add(tf.matMul(tf.tensor2d([[2, 4]]), projection), bias);
      const modified = tf.add(tf.matMul(tf.tensor2d([[1, 4]]), projection), bias);
      const delta = tf.matMul(tf.tensor2d([[-1, 0]]), projection);
      return { baseline: Array.from(baseline.dataSync()), modified: Array.from(modified.dataSync()), delta: Array.from(delta.dataSync()) };
    });
    expect(result).toEqual({ baseline: [10.5, 9.5], modified: [9.5, 6.5], delta: [-1, -3] });
  });

  it('matches every exported stage for all 84 intervention fixtures', () => {
    const model = new TransformerModel(reference.manifest.config, reference.weights);
    let maximum = 0;
    try {
      for (const example of reference.cases) {
        const result = model.forward(example.tokenIds, example.interventions as HeadScaleIntervention[], references(Object.keys(example.activations)));
        try {
          for (const [key, expected] of Object.entries(example.activations)) {
            expect(result.captured[key].shape, `${example.name}/${key}`).toEqual(expected.shape);
            maximum = Math.max(maximum, assertClose(result.captured[key].dataSync(), expected.values));
          }
        } finally { disposeForward(result); }
      }
      console.info(`CPU forward maximum absolute error across 84 fixtures: ${maximum}`);
    } finally { model.dispose(); }
  });

  it('matches loss, every parameter gradient, and a real Adam update', () => {
    const model = new TransformerModel(reference.manifest.config, reference.weights);
    const settings = reference.training.optimizer;
    const optimizer = tf.train.adam(settings.learningRate, settings.beta1, settings.beta2, settings.epsilon);
    const { tokenIds, targets } = reference.training;
    const gradient = tf.variableGrads(() => model.loss(tokenIds, targets), Object.values(model.weights));
    let maximum = 0, updateError = 0;
    try {
      assertClose(gradient.value.dataSync(), [reference.training.loss]);
      for (const [name, variable] of Object.entries(model.weights))
        maximum = Math.max(maximum, assertClose(gradient.grads[variable.name].dataSync(), reference.training.gradients[name as keyof typeof reference.training.gradients].values, 3e-5, 3e-4));
      optimizer.applyGradients(Object.entries(gradient.grads).map(([name, tensor]) => ({ name, tensor })));
      for (const [name, variable] of Object.entries(model.weights)) {
        try { updateError = Math.max(updateError, assertClose(variable.dataSync(), reference.training.updatedWeights[name as keyof typeof reference.training.updatedWeights].values, 3e-5, 1e-5)); }
        catch (error) { throw new Error(`${name}: ${String(error)}; gradients=${Array.from(gradient.grads[variable.name].dataSync())}`); }
      }
      console.info(`CPU gradient maximum absolute error: ${maximum}; Adam update: ${updateError}`);
    } finally {
      gradient.value.dispose(); Object.values(gradient.grads).forEach(value => value.dispose()); optimizer.dispose(); model.dispose();
    }
  });

  it('agrees with central finite differences for selected parameters', () => {
    const model = new TransformerModel(reference.manifest.config, reference.weights);
    const { tokenIds, targets } = reference.training;
    const gradient = tf.variableGrads(() => model.loss(tokenIds, targets), Object.values(model.weights));
    try {
      for (const [name, index] of [['tokenEmbedding', 8], ['blocks.0.q.weight', 3], ['blocks.1.mlpIn.weight', 17], ['finalNorm.scale', 2]] as const) {
        const variable = model.weights[name];
        const original = Float32Array.from(variable.dataSync());
        const evaluate = (offset: number) => tf.tidy(() => {
          const modified = original.slice(); modified[index] += offset;
          variable.assign(tf.tensor(modified, variable.shape));
          return model.loss(tokenIds, targets).dataSync()[0];
        });
        const delta = .001;
        const numerical = (evaluate(delta) - evaluate(-delta)) / (2 * delta);
        tf.tidy(() => variable.assign(tf.tensor(original, variable.shape)));
        assertClose([numerical], [gradient.grads[variable.name].dataSync()[index]], 7e-4, 3e-3);
      }
    } finally { gradient.value.dispose(); Object.values(gradient.grads).forEach(g => g.dispose()); model.dispose(); }
  });
});

describe('causality, validation, and resource ownership', () => {
  it('does not alter earlier predictions or propagate an earlier final-layer edit to the final position', () => {
    const model = new TransformerModel(reference.manifest.config, reference.weights);
    const tokens = [1, 5, 9, 3, 7];
    const baseline = model.forward(tokens);
    const late: HeadScaleIntervention = { id: 'late', kind: 'head.scale', layer: 0, head: 0, factor: 0, tokens: { kind: 'indices', indices: [3] } };
    const earlier = { ...late, layer: 1, tokens: { kind: 'indices' as const, indices: [0] } };
    const modifiedLate = model.forward(tokens, [late]);
    const modifiedEarlier = model.forward(tokens, [earlier]);
    try {
      const original = baseline.logits.dataSync();
      assertClose(modifiedLate.logits.dataSync().slice(0, 3 * 16), original.slice(0, 3 * 16), 0, 0);
      assertClose(modifiedEarlier.logits.dataSync().slice(4 * 16), original.slice(4 * 16), 0, 0);
      expect(canReachPrediction(earlier, 4, model.config, tokens.length)).toBe(false);
      expect(canReachPrediction(late, 4, model.config, tokens.length)).toBe(true);
    } finally { disposeForward(baseline); disposeForward(modifiedLate); disposeForward(modifiedEarlier); model.dispose(); }
  });

  it('keeps tensor counts bounded, including rejected requests', () => {
    const start = tf.memory().numTensors;
    const model = new TransformerModel(reference.manifest.config, reference.weights);
    const loaded = tf.memory().numTensors;
    const patch: HeadScaleIntervention = { id: 'test', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'all' } };
    try {
      for (let i = 0; i < 50; i++) {
        const result = model.forward([1, 5, 9], [patch], [{ stage: 'attention.head.raw', layer: 0 }]);
        disposeForward(result);
        expect(() => model.forward([1, 5, 9], [{ ...patch, factor: NaN }])).toThrow();
        expect(tf.memory().numTensors).toBe(loaded);
      }
    } finally { model.dispose(); model.dispose(); }
    expect(tf.memory().numTensors).toBe(start);
  });

  it('rejects overlaps, duplicate tokens, malformed weights, and out-of-range inputs', () => {
    const model = new TransformerModel(reference.manifest.config, reference.weights);
    const patch: HeadScaleIntervention = { id: 'test', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'indices', indices: [0, 0] } };
    try {
      expect(() => model.forward([1], [patch])).toThrow(/overlap|duplicate/);
      expect(() => model.forward([1], [{ ...patch, tokens: { kind: 'all' } }, { ...patch, tokens: { kind: 'all' } }])).toThrow(/overlap/);
      expect(() => model.forward([])).toThrow(); expect(() => model.forward([16])).toThrow();
      expect(() => model.forward(Array(9).fill(0))).toThrow();
      const corrupted: Record<string, TensorRecord> = { ...reference.weights, tokenEmbedding: { shape: [16, 8], values: [NaN] } };
      expect(() => new TransformerModel(reference.manifest.config, corrupted)).toThrow(/Invalid values/);
    } finally { model.dispose(); }
  });
});

describe('artifact and comparison contracts', () => {
  it('validates the manifest and rejects missing or incompatible conventions', () => {
    expect(validateManifest(reference.manifest).tensors.length).toBe(36);
    expect(() => validateManifest({ ...reference.manifest, conventions: { ...reference.manifest.conventions, gelu: 'exact' } })).toThrow();
    expect(() => validateManifest({ ...reference.manifest, tensors: [] })).toThrow();
    const bad = structuredClone(reference.manifest); bad.tensors[0].byteOffset = 4;
    expect(() => validateManifest(bad)).toThrow();
  });

  const tokenizer = { schemaVersion: 1, kind: 'symbols', bosTokenId: 0, tokens: ['<BOS>', 'A', 'B'] } as const;
  it('tokenizes without silent truncation or symbol normalization', () => {
    const t = validateTokenizer({ ...tokenizer, tokens: [...tokenizer.tokens] }, 3);
    expect(tokenize(' A  B ', t, 3)).toEqual([0, 1, 2]); expect(tokenize('', t, 3)).toEqual([0]);
    expect(() => tokenize('a', t, 3)).toThrow(/Unknown symbol/);
    expect(() => tokenize('A B A', t, 3)).toThrow(/at most/);
  });
  it('uses the complete distribution and stable logarithms', () => {
    const t = validateTokenizer({ ...tokenizer, tokens: [...tokenizer.tokens] }, 3);
    const result = comparePredictions([1000, 0, -1000], [1001, 1, -999], t, 1, 2);
    expect(result.metrics.jsDivergence).toBeCloseTo(0, 12);
    expect(result.metrics.maxAbsoluteLogitChange).toBe(1);
    expect(result.metrics.targetLogProbabilityDelta).toBe(0);
    const changed = comparePredictions([1, 2, 3], [3, 2, 1], t, 1, 2);
    expect(changed.metrics.jsDivergence).toBeGreaterThan(0);
    expect(changed.metrics.targetLogProbabilityDelta).toBeLessThan(0);
    expect(changed.displayTokens.map(token => token.tokenId)).toEqual(expect.arrayContaining([0, 2]));
  });
});
