import { beforeAll, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import reference from '../fixtures/reference.json';
import suffixReference from '../fixtures/suffix-reference.json';
import { TransformerModel, disposeForward, type ForwardResult } from '../../src/engine/model';
import { planExecution } from '../../src/engine/execution';
import type { ActivationReference, HeadScaleIntervention, ModelConfig, TensorRecord } from '../../src/engine/contracts';
import { assertClose } from './numerical';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
const patch: HeadScaleIntervention = { id: 'scale', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'all' } };

describe('pure execution planner', () => {
  it('recognizes effective edits without letting identity patches force a slower plan', () => {
    expect(planExecution([], false).strategy).toBe('identity');
    expect(planExecution([{ ...patch, factor: 1 }], true).strategy).toBe('identity');
    expect(planExecution([{ ...patch, tokens: { kind: 'indices', indices: [] } }], true).strategy).toBe('identity');
    expect(planExecution([patch, { ...patch, layer: 1, factor: 1 }], true)).toMatchObject({ strategy: 'cached-suffix', fromLayer: 0 });
    expect(planExecution([patch], false).strategy).toBe('full');
    expect(planExecution([patch, { ...patch, head: 1 }], true).strategy).toBe('cached-suffix');
    expect(planExecution([patch, { ...patch, layer: 1 }], true).strategy).toBe('full');
    expect(planExecution([], true, 'full').strategy).toBe('full');
  });
});

it('matches independent residuals and all-position logits in 319 cached fixtures', () => {
  const start = tf.memory().numTensors;
  let cachedCount = 0, fullCount = 0, logitError = 0, residualError = 0, cachedFullError = 0;
  const legacy = { config: reference.manifest.config, weights: reference.weights, cases: reference.cases.map(example => ({
    name: example.name, tokenIds: example.tokenIds, interventions: example.interventions,
    logits: example.activations.logits,
    residuals: Object.fromEntries([0, 1].map(layer => [String(layer), example.activations[`${layer}:attention.residual` as keyof typeof example.activations]])),
  })) };
  const verify = (config: ModelConfig, weights: Record<string, TensorRecord>, cases: {
    name: string; tokenIds: number[]; interventions: unknown; logits: TensorRecord; residuals: Record<string, TensorRecord>;
  }[]) => {
    const model = new TransformerModel(config, weights);
    const baselines = new Map<string, ForwardResult>();
    const capture: ActivationReference[] = Array.from({ length: config.layers }, (_, layer) => [
      { layer, stage: 'attention.head.raw' as const }, { layer, stage: 'attention.residual' as const },
    ]).flat();
    try {
      for (const example of cases) {
        const patches = example.interventions as HeadScaleIntervention[];
        const full = model.forward(example.tokenIds, patches);
        try {
          const expected = example.logits.values;
          logitError = Math.max(logitError, assertClose(full.logits.dataSync(), expected, `${example.name}/full`));
          const layers = new Set(patches.map(p => p.layer));
          if (layers.size !== 1) { fullCount++; continue; }
          const layer = patches[0].layer;
          const key = JSON.stringify(example.tokenIds);
          let baseline = baselines.get(key);
          if (!baseline) { baseline = model.forward(example.tokenIds, [], capture); baselines.set(key, baseline); }
          const cached = model.forwardCached({
            rawHeadOutputs: baseline.captured[`${layer}:attention.head.raw`] as tf.Tensor4D,
            attentionResidual: baseline.captured[`${layer}:attention.residual`] as tf.Tensor3D,
          }, layer, patches, [{ layer, stage: 'attention.residual' }]);
          try {
            cachedCount++;
            logitError = Math.max(logitError, assertClose(cached.logits.dataSync(), expected, `${example.name}/cached`));
            cachedFullError = Math.max(cachedFullError, assertClose(cached.logits.dataSync(), full.logits.dataSync(), `${example.name}/equivalence`));
            residualError = Math.max(residualError, assertClose(cached.captured[`${layer}:attention.residual`].dataSync(), example.residuals[String(layer)].values, `${example.name}/residual`));
          } finally { disposeForward(cached); }
        } finally { disposeForward(full); }
      }
    } finally { baselines.forEach(disposeForward); model.dispose(); }
  };
  verify(legacy.config, legacy.weights, legacy.cases);
  for (const model of suffixReference.models) verify(model.config, model.weights, model.cases);
  expect(cachedCount).toBe(319);
  expect(fullCount).toBe(26);
  expect(tf.memory().numTensors).toBe(start);
  console.info(JSON.stringify({ cachedCount, fullCount, logitError, residualError, cachedFullError }));
});

it('rejects invalid cached shapes and cross-layer edits before tensor allocation', () => {
  const model = new TransformerModel(reference.manifest.config, reference.weights);
  const baseline = model.forward([1, 5], [], [{ layer: 0, stage: 'attention.head.raw' }, { layer: 0, stage: 'attention.residual' }]);
  const cache = { rawHeadOutputs: baseline.captured['0:attention.head.raw'] as tf.Tensor4D,
    attentionResidual: baseline.captured['0:attention.residual'] as tf.Tensor3D };
  try {
    const count = tf.memory().numTensors;
    expect(() => model.forwardCached(cache, 0, [patch, { ...patch, layer: 1 }])).toThrow(/one layer/);
    expect(() => model.forwardCached(cache, 2, [patch])).toThrow(/Invalid cached layer/);
    expect(tf.memory().numTensors).toBe(count);
    const wrongResidual = tf.zeros([1, 3, 8]) as tf.Tensor3D;
    try {
      expect(() => model.forwardCached({ ...cache, attentionResidual: wrongResidual }, 0, [patch])).toThrow(/Invalid cached tensor shapes/);
      expect(tf.memory().numTensors).toBe(count + 1);
    } finally { wrongResidual.dispose(); }
    const owned = model.inspectCached(cache, 0, [], 'attention.head.raw')!;
    expect(owned).not.toBe(cache.rawHeadOutputs); owned.dispose();
    expect(cache.rawHeadOutputs.isDisposed).toBe(false);
  } finally { disposeForward(baseline); model.dispose(); }
});
