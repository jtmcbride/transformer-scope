import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import reference from '../fixtures/reference.json';
import { RunEngine } from '../../src/engine/runs';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import { type ModelArtifact, type HeadScaleIntervention } from '../../src/engine/contracts';

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
