import { beforeAll, expect, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import { MathBackendCPU } from '@tensorflow/tfjs-backend-cpu';
import reference from '../fixtures/reference.json';
import { RunEngine } from '../../src/engine/runs';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import { type ModelArtifact, type HeadScaleIntervention } from '../../src/engine/contracts';
import { assertClose } from './numerical';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
function artifact(): ModelArtifact {
  return { manifest: validateManifest(reference.manifest), weights: reference.weights,
    tokenizer: validateTokenizer({ schemaVersion: 1, kind: 'symbols', bosTokenId: 0, tokens: ['<BOS>', ...'ABCDEFGHIJKLMNO'] }, 16) };
}
const patch: HeadScaleIntervention = { id: 'scale', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'all' } };

it('inspects immutable modified runs even after releasing their baseline', async () => {
  const before = tf.memory().numTensors;
  const engine = new RunEngine(artifact());
  try {
    const input = [1, 5, 9, 3, 7];
    const baseline = await engine.runBaseline(input);
    input[0] = 15;
    const submitted = structuredClone(patch);
    const comparison = await engine.compare({ baselineRunId: baseline.runId, interventions: [submitted], predictionPosition: 4 });
    submitted.factor = 2;
    const raw = await engine.inspectActivation(baseline.runId, { layer: 0, stage: 'attention.head.raw' }, { head: 0, position: 4 });
    engine.releaseRun(baseline.runId); engine.releaseRun(baseline.runId);
    const modified = await engine.inspectActivation(comparison.modifiedRunId, { layer: 0, stage: 'attention.head.modified' }, { head: 0, position: 4 });
    expect(modified.shape).toEqual([1, 1, 1, 4]);
    modified.values.forEach((value, i) => expect(value).toBeCloseTo(raw.values[i] * .5, 6));
    expect(comparison.interventions[0].factor).toBe(.5);
    await expect(engine.inspectActivation(baseline.runId, { stage: 'logits' })).rejects.toThrow(/unavailable/);
    engine.releaseRun(comparison.modifiedRunId);
    await expect(engine.inspectActivation(comparison.modifiedRunId, { stage: 'logits' })).rejects.toThrow(/unavailable/);
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(before);
});

it('restores the baseline through an identity comparison and rejects invalid selections without leaks', async () => {
  const engine = new RunEngine(artifact());
  try {
    const baseline = await engine.runBaseline([1, 5, 9]);
    const loaded = tf.memory().numTensors;
    const identity = await engine.compare({ baselineRunId: baseline.runId, interventions: [{ ...patch, factor: 1 }], targetTokenId: 9 });
    expect(identity.execution.strategy).toBe('identity');
    expect(identity.metrics).toEqual({ jsDivergence: 0, maxAbsoluteLogitChange: 0, targetLogProbabilityDelta: 0 });
    await expect(engine.inspectActivation(baseline.runId, { layer: 0, stage: 'attention.weights' }, { head: 2 })).rejects.toThrow(/Invalid head/);
    await expect(engine.inspectActivation(baseline.runId, { stage: 'logits' }, { nonsense: 0 })).rejects.toThrow(/unknown tensor axis/);
    await expect(engine.inspectActivation(baseline.runId, { layer: 0, stage: 'logits' })).rejects.toThrow(/invalid layer/);
    expect(tf.memory().numTensors).toBe(loaded);
  } finally { engine.dispose(); }
});

it('bounds live descriptors and releases failed comparisons safely', async () => {
  const engine = new RunEngine(artifact());
  try {
    const baseline = await engine.runBaseline([1, 5]);
    for (let i = 0; i < 63; i++) await engine.compare({ baselineRunId: baseline.runId, interventions: [] });
    const count = tf.memory().numTensors;
    await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [patch] })).rejects.toThrow(/Release an older run/);
    expect(tf.memory().numTensors).toBe(count);
    engine.dispose();
    await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [] })).rejects.toThrow(/session has ended/);
  } finally { engine.dispose(); }
});

it('keeps cached/full inspection equivalent across upstream, edited, and downstream stages', async () => {
  const engine = new RunEngine(artifact());
  try {
    const baseline = await engine.runBaseline([1, 5, 9, 3, 7]);
    const request = { baselineRunId: baseline.runId, interventions: [patch], targetTokenId: 9, topK: 16 };
    const cached = await engine.compare(request);
    const full = await engine.compare({ ...request, executionPreference: 'full' });
    expect(cached.execution).toMatchObject({ strategy: 'cached-suffix', recomputedFromLayer: 0 });
    expect(full.execution.strategy).toBe('full');
    for (const reference of [
      { stage: 'embedding.output' as const }, { stage: 'logits' as const }, { stage: 'final.normalized' as const },
      ...[0, 1].flatMap(layer => ['block.input', 'attention.query', 'attention.weights', 'attention.head.raw',
        'attention.head.modified', 'attention.projected', 'attention.residual', 'mlp.output', 'block.output'].map(stage =>
          ({ layer, stage: stage as Parameters<RunEngine['inspectActivation']>[1]['stage'] }))),
    ]) {
      const a = await engine.inspectActivation(cached.modifiedRunId, reference);
      const b = await engine.inspectActivation(full.modifiedRunId, reference);
      assertClose(a.values, b.values, JSON.stringify(reference));
    }
    const forcedIdentity = await engine.compare({ ...request, interventions: [{ ...patch, factor: 1 }], executionPreference: 'full' });
    expect(forcedIdentity.execution.strategy).toBe('full');
    expect(forcedIdentity.metrics.maxAbsoluteLogitChange).toBe(0);
    const cross = await engine.compare({ ...request, interventions: [patch, { ...patch, layer: 1, head: 1 }] });
    expect(cross.execution.strategy).toBe('full');
    const original = await engine.inspectActivation(baseline.runId, { stage: 'logits' });
    const restored = await engine.compare({ ...request, interventions: [{ ...patch, factor: 1 }] });
    const identity = await engine.inspectActivation(restored.modifiedRunId, { stage: 'logits' });
    assertClose(identity.values, original.values, 'restored baseline', 0, 0);
  } finally { engine.dispose(); }
});

it('falls back after LRU eviction and replays modified runs after baseline cache release', async () => {
  const engine = new RunEngine(artifact(), { cacheBudgetBytes: 640 });
  try {
    const a = await engine.runBaseline([1, 5, 9, 3, 7]);
    const first = await engine.compare({ baselineRunId: a.runId, interventions: [patch] });
    expect(first.execution.strategy).toBe('cached-suffix');
    const before = await engine.inspectActivation(first.modifiedRunId, { stage: 'logits' });
    const b = await engine.runBaseline([2, 5, 9, 3, 7]);
    expect(engine.cacheStats()).toMatchObject({ logicalBytes: 640, baselines: 1 });
    const fallback = await engine.compare({ baselineRunId: a.runId, interventions: [patch] });
    expect(fallback.execution.strategy).toBe('full');
    const after = await engine.inspectActivation(first.modifiedRunId, { stage: 'logits' });
    assertClose(after.values, before.values, 'replay after eviction');
    expect((await engine.compare({ baselineRunId: b.runId, interventions: [patch] })).execution.strategy).toBe('cached-suffix');
    engine.releaseRun(a.runId); engine.releaseRun(b.runId);
    expect(engine.cacheStats().logicalBytes).toBe(0);
    assertClose((await engine.inspectActivation(first.modifiedRunId, { stage: 'logits' })).values, before.values, 'replay after release');
  } finally { engine.dispose(); }
});

it('supports a zero cache budget without changing the result', async () => {
  const engine = new RunEngine(artifact(), { cacheBudgetBytes: 0 });
  try {
    const baseline = await engine.runBaseline([1, 5, 9, 3, 7]);
    expect(engine.cacheStats().logicalBytes).toBe(0);
    const comparison = await engine.compare({ baselineRunId: baseline.runId, interventions: [patch] });
    expect(comparison.execution.strategy).toBe('full');
    const actual = await engine.inspectActivation(comparison.modifiedRunId, { stage: 'logits' });
    const expected = reference.cases.find(example => example.name === 'length-5-case-2')!;
    assertClose(actual.values, expected.activations.logits.values, 'zero-budget fallback');
  } finally { engine.dispose(); }
});

function holdNextRead() {
  let resume!: () => void;
  const original = tf.Tensor.prototype.data;
  const spy = vi.spyOn(tf.Tensor.prototype, 'data').mockImplementationOnce(function (this: tf.Tensor) {
    const read = original.call(this);
    return new Promise((resolve, reject) => { resume = () => { void read.then(resolve, reject); }; });
  });
  return { resume: () => resume(), restore: () => spy.mockRestore() };
}

it('pins a released baseline until an in-flight comparison completes', async () => {
  const before = tf.memory().numTensors;
  const engine = new RunEngine(artifact(), { cacheBudgetBytes: 640 });
  const baseline = await engine.runBaseline([1, 5, 9, 3, 7]);
  const gate = holdNextRead();
  const pending = engine.compare({ baselineRunId: baseline.runId, interventions: [patch] });
  try {
    expect(engine.cacheStats().pinnedEntries).toBe(1);
    engine.releaseRun(baseline.runId);
    const another = await engine.runBaseline([2, 5, 9, 3, 7]);
    expect(engine.cacheStats()).toMatchObject({ logicalBytes: 640, baselines: 0, pinnedEntries: 1, retiredEntries: 1 });
    gate.resume();
    const comparison = await pending;
    expect(comparison.execution.strategy).toBe('cached-suffix');
    expect(engine.cacheStats().logicalBytes).toBe(0);
    expect((await engine.compare({ baselineRunId: another.runId, interventions: [patch] })).execution.strategy).toBe('full');
  } finally { gate.resume(); gate.restore(); await pending.catch(() => {}); engine.dispose(); }
  expect(tf.memory().numTensors).toBe(before);
});

it('defers model disposal during an in-flight read and releases every resource afterward', async () => {
  const before = tf.memory().numTensors;
  const engine = new RunEngine(artifact());
  const baseline = await engine.runBaseline([1, 5, 9]);
  const gate = holdNextRead();
  const pending = engine.compare({ baselineRunId: baseline.runId, interventions: [patch] });
  try {
    engine.dispose();
    expect(engine.cacheStats().retiredEntries).toBe(1);
    gate.resume();
    await expect(pending).rejects.toThrow(/session has ended/);
    expect(engine.cacheStats().logicalBytes).toBe(0);
  } finally { gate.resume(); gate.restore(); await pending.catch(() => {}); engine.dispose(); }
  expect(tf.memory().numTensors).toBe(before);
});

it('releases leases and temporaries on failed readback', async () => {
  const engine = new RunEngine(artifact());
  try {
    const baseline = await engine.runBaseline([1, 5, 9]);
    const count = tf.memory().numTensors;
    const spy = vi.spyOn(tf.Tensor.prototype, 'data').mockRejectedValueOnce(new Error('Readback failed'));
    try { await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [patch] })).rejects.toThrow(/Readback failed/); }
    finally { spy.mockRestore(); }
    expect(engine.cacheStats().pinnedEntries).toBe(0);
    expect(tf.memory().numTensors).toBe(count);
    expect((await engine.compare({ baselineRunId: baseline.runId, interventions: [patch] })).execution.strategy).toBe('cached-suffix');
  } finally { engine.dispose(); }
});

it('rejects numerical failures without admitting a baseline cache or leaking captured tensors', async () => {
  const before = tf.memory().numTensors;
  const bad = artifact();
  bad.weights = structuredClone(bad.weights);
  bad.weights['blocks.0.q.weight'].values[0] = 3e38;
  bad.weights['blocks.0.k.weight'].values[0] = 3e38;
  const engine = new RunEngine(bad);
  try {
    const loaded = tf.memory().numTensors;
    await expect(engine.runBaseline([1, 5, 9])).rejects.toThrow(/non-finite logits/);
    expect(engine.cacheStats().logicalBytes).toBe(0);
    expect(tf.memory().numTensors).toBe(loaded);
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(before);
});

it('snapshots inputs and request identity before asynchronous readback', async () => {
  const engine = new RunEngine(artifact());
  try {
    const input = [1, 5, 9];
    const pending = engine.runBaseline(input); input[0] = 15;
    const baseline = await pending;
    expect(baseline.tokenIds).toEqual([1, 5, 9]);
    const request = { baselineRunId: baseline.runId, interventions: [structuredClone(patch)] };
    const comparison = engine.compare(request);
    request.baselineRunId = 'changed'; request.interventions[0].factor = 2;
    const result = await comparison;
    expect(result.baselineRunId).toBe(baseline.runId);
    expect(result.interventions[0].factor).toBe(.5);
  } finally { engine.dispose(); }
});

it('invalidates the session on backend changes and rejects old checkpoint/runtime handles', async () => {
  const before = tf.memory().numTensors;
  const engine = new RunEngine(artifact());
  const baseline = await engine.runBaseline([1, 5, 9]);
  tf.registerBackend('cache-test-cpu', () => new MathBackendCPU());
  try {
    await tf.setBackend('cache-test-cpu');
    await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [patch] })).rejects.toThrow(/backend changed/);
    expect(engine.cacheStats().logicalBytes).toBe(0);
  } finally { engine.dispose(); await tf.setBackend('cpu'); tf.removeBackend('cache-test-cpu'); }
  const checkpoint = artifact(); checkpoint.manifest.modelRevision = 'f'.repeat(64);
  const other = new RunEngine(checkpoint);
  try { await expect(other.compare({ baselineRunId: baseline.runId, interventions: [patch] })).rejects.toThrow(/another model session/); }
  finally { other.dispose(); }
  expect(tf.memory().numTensors).toBe(before);
});

it('retains bounded tensors across 1,000 cached interventions, resets, failures, and inspections', async () => {
  const before = tf.memory().numTensors;
  const engine = new RunEngine(artifact());
  try {
    const baseline = await engine.runBaseline([1, 5, 9, 3, 7]);
    const loaded = tf.memory().numTensors;
    for (let i = 0; i < 1000; i++) {
      const factor = [0, .5, 1, 2][i % 4];
      const result = await engine.compare({ baselineRunId: baseline.runId, interventions: [{ ...patch, factor, layer: i % 2 }] });
      expect(result.execution.strategy).toBe(factor === 1 ? 'identity' : 'cached-suffix');
      if (i % 25 === 0) {
        await engine.inspectActivation(result.modifiedRunId, { layer: i % 2, stage: 'attention.residual' });
        await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [{ ...patch, factor: NaN }] })).rejects.toThrow();
      }
      engine.releaseRun(result.modifiedRunId);
      expect(tf.memory().numTensors).toBe(loaded);
      expect(engine.cacheStats().pinnedEntries).toBe(0);
    }
    engine.releaseRun(baseline.runId);
    expect(engine.cacheStats().logicalBytes).toBe(0);
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(before);
});
