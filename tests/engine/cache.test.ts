import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { BaselineCache } from '../../src/engine/cache';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
function layers(value: number) {
  return tf.tidy(() => [{ rawHeadOutputs: tf.fill([1, 2, 2, 2], value) as tf.Tensor4D,
    attentionResidual: tf.fill([1, 2, 4], value) as tf.Tensor3D }]);
}

it('evicts the least recently used unpinned baseline and invalidates mismatched identities', () => {
  const before = tf.memory().numTensors;
  const cache = new BaselineCache(128);
  const a = layers(1), b = layers(2), c = layers(3);
  try {
    cache.put('a', 'model-a', a); cache.put('b', 'model-b', b);
    const touch = cache.acquire('a', 'model-a')!; touch.release();
    cache.put('c', 'model-c', c);
    expect(b[0].rawHeadOutputs.isDisposed).toBe(true);
    expect(a[0].rawHeadOutputs.dataSync()[0]).toBe(1);
    expect(cache.stats().logicalBytes).toBe(128);
    expect(cache.acquire('c', 'different-model')).toBeUndefined();
    expect(c[0].rawHeadOutputs.isDisposed).toBe(true);
  } finally { cache.clear(); }
  expect(tf.memory().numTensors).toBe(before);
});

it('holds retired entries through read leases and rejects candidates while all capacity is pinned', () => {
  const before = tf.memory().numTensors;
  const cache = new BaselineCache(64);
  const a = layers(1);
  cache.put('a', 'identity', a);
  const lease = cache.acquire('a', 'identity')!;
  try {
    cache.delete('a'); cache.clear();
    expect(cache.stats()).toMatchObject({ logicalBytes: 64, baselines: 0, pinnedEntries: 1, retiredEntries: 1 });
    const rejected = layers(2);
    expect(cache.put('b', 'identity', rejected)).toBe(false);
    expect(rejected[0].rawHeadOutputs.isDisposed).toBe(true);
    expect(lease.layers[0].rawHeadOutputs.dataSync()[0]).toBe(1);
  } finally { lease.release(); lease.release(); cache.clear(); }
  expect(cache.stats().logicalBytes).toBe(0);
  expect(tf.memory().numTensors).toBe(before);
});

it('takes ownership of oversized candidates and validates budgets', () => {
  const before = tf.memory().numTensors;
  expect(() => new BaselineCache(-1)).toThrow();
  expect(() => new BaselineCache(NaN)).toThrow();
  const cache = new BaselineCache(0), candidate = layers(1);
  expect(cache.put('a', 'identity', candidate)).toBe(false);
  expect(candidate[0].rawHeadOutputs.isDisposed).toBe(true);
  expect(tf.memory().numTensors).toBe(before);
});
