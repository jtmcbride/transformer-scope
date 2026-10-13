import { beforeAll, expect, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import { TransformerModel } from '../../src/engine/model';
import { checkTracePrediction, measureTrace, traceBudget, TRACE_CAPTURE_BUDGET } from '../../src/engine/trace';
import { traceNodeKey, type TraceAddress } from '../../src/engine/trace-topology';
import { normalizedAttention, type AttentionOverrideIntervention, type HeadScaleIntervention } from '../../src/engine/contracts';
import fixture from '../fixtures/trace-reference.json';
import redirectFixture from '../fixtures/attention-trace-reference.json';
import { loadShippedArtifact } from './retrieval-artifact';
import { assertClose } from './numerical';
import { firstReadoutFlip, readoutTitle } from '../../src/trace-view';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
const retrieval = loadShippedArtifact('retrieval'), verification = loadShippedArtifact('verification');

it('redirect traces match independent rows, stages and readouts without editing source positions or retaining tensors', async () => {
  const initial = tf.memory().numTensors;
  for (const item of redirectFixture.cases) {
    const loaded = item.modelRevision === retrieval.manifest.modelRevision ? retrieval : verification;
    const engine = new RunEngine(loaded.artifact), patch = item.intervention as AttentionOverrideIntervention;
    try {
      const baseline = await engine.runBaseline(item.tokens);
      const comparison = await engine.compare({ baselineRunId: baseline.runId, interventions: [patch], predictionPosition: item.predictionPosition });
      const count = tf.memory().numTensors, cache = engine.cacheStats();
      const trace = await engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, item.predictionPosition);
      expect(trace.hasComputationalPath).toBe(comparison.hasComputationalPath);
      expect(trace.nodes).toHaveLength(item.nodes.length);
      for (const [i, node] of trace.nodes.entries()) {
        const expected = item.nodes[i];
        expect(node.key).toBe(traceNodeKey(expected as TraceAddress));
        assertClose([node.baselineNorm, node.modifiedNorm, node.maxDifference, node.differenceNorm],
          [expected.baselineNorm, expected.modifiedNorm, expected.maxDifference, expected.differenceNorm], `${item.name} ${node.key}`, .0005, 2e-5);
        if (node.stage === 'embedding.output' || node.layer === patch.layer &&
          (['block.input', 'attention.normalized'].includes(node.stage) || node.position !== patch.query || node.head !== undefined && node.head !== patch.head)) expect(node.change, node.key).toBe('exact');
      }
      assertClose(trace.attentionRow!.baselineWeights, item.attentionRow.baselineWeights, `${item.name} baseline row`);
      expect(trace.attentionRow!.modifiedWeights).toEqual(Array.from(normalizedAttention(patch.weights)));
      assertClose(trace.attentionRow!.modifiedWeights, item.attentionRow.modifiedWeights, `${item.name} modified row`);
      const scores = measureTrace(item.attentionRow.baselineScores, item.attentionRow.modifiedScores);
      assertClose([trace.attentionRow!.scores.baselineNorm, trace.attentionRow!.scores.modifiedNorm], [scores.baselineNorm, scores.modifiedNorm], `${item.name} scores`);
      expect(trace.attentionRow!.scores.change).toBe('exact');
      item.baselineReadouts.forEach((row, i) => {
        assertClose(trace.readouts[i].baselineLogits, row, `${item.name} lens ${i} baseline`);
        assertClose(trace.readouts[i].modifiedLogits, item.modifiedReadouts[i], `${item.name} lens ${i} modified`);
      });
      assertClose(trace.readouts.at(-1)!.baselineLogits, item.baselineLogits, `${item.name} prediction baseline`);
      assertClose(trace.readouts.at(-1)!.modifiedLogits, item.modifiedLogits, `${item.name} prediction modified`);
      if (item.name === 'redirect-effect') expect(readoutTitle(firstReadoutFlip(trace)!)).toBe('Layer 2 after attention');
      if (['redirect-future', 'redirect-unreachable'].includes(item.name)) {
        expect(trace.hasComputationalPath).toBe(false);
        expect(trace.nodes.some(node => node.route === 'local' && node.change !== 'exact')).toBe(true);
        expect(trace.nodes.find(node => node.stage === 'logits' && node.position === item.predictionPosition)?.change).toBe('exact');
      }
      if (item.name === 'redirect-identical-row') expect(trace.nodes.every(node => node.change === 'exact')).toBe(true);
      for (let i = 0; i < 3; i++) await engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, item.predictionPosition);
      expect(tf.memory().numTensors).toBe(count);
      expect(engine.cacheStats()).toEqual(cache);
    } finally { engine.dispose(); }
  }
  expect(tf.memory().numTensors).toBe(initial);
});

it('all seven traces match independent PyTorch stages and readouts and retain no tensors', async () => {
  const initial = tf.memory().numTensors;
  for (const item of fixture.cases) {
    const loaded = ['effect', 'small', 'unreachable'].includes(item.name) ? retrieval : verification;
    expect(item.modelRevision).toBe(loaded.manifest.modelRevision);
    const engine = new RunEngine(loaded.artifact);
    try {
      const baseline = await engine.runBaseline(item.tokens);
      const comparison = await engine.compare({ baselineRunId: baseline.runId, interventions: [item.intervention as HeadScaleIntervention], predictionPosition: item.predictionPosition });
      const count = tf.memory().numTensors, cache = engine.cacheStats();
      const trace = await engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, item.predictionPosition);
      expect(trace.hasComputationalPath).toBe(comparison.hasComputationalPath);
      expect(trace.nodes).toHaveLength(item.nodes.length);
      const expected = new Map(item.nodes.map(node => [traceNodeKey(node as TraceAddress), node]));
      for (const node of trace.nodes) {
        const reference = expected.get(node.key)!;
        assertClose([node.baselineNorm, node.modifiedNorm, node.maxDifference, node.differenceNorm],
          [reference.baselineNorm, reference.modifiedNorm, reference.maxDifference, reference.differenceNorm], `${item.name} ${node.key}`, .0005, 2e-5);
        if (node.stage === 'embedding.output' || node.layer === item.intervention.layer && ['attention.weights', 'attention.head.raw'].includes(node.stage)) expect(node.change, node.key).toBe('exact');
        if (item.name === 'identity') expect(node.change).toBe('exact');
      }
      item.baselineReadouts.forEach((row, i) => {
        assertClose(trace.readouts[i].baselineLogits, row, `${item.name} lens ${i} baseline`);
        assertClose(trace.readouts[i].modifiedLogits, item.modifiedReadouts[i], `${item.name} lens ${i} modified`);
      });
      assertClose(trace.readouts.at(-1)!.baselineLogits, item.baselineLogits, `${item.name} prediction baseline`);
      assertClose(trace.readouts.at(-1)!.modifiedLogits, item.modifiedLogits, `${item.name} prediction modified`);
      expect(trace.readouts.at(-1)!.prediction).toBe(true);
      expect(trace.readouts.slice(0, -1).every(row => !row.prediction)).toBe(true);
      if (item.name === 'effect') expect(readoutTitle(firstReadoutFlip(trace)!)).toBe('Layer 2 after attention');
      if (item.name === 'small') {
        expect(trace.nodes.some(node => node.position === item.predictionPosition && node.route === 'prediction' && node.maxDifference > 0)).toBe(true);
        // This recipe has a tiny probability change despite noticeable vector differences.
        // Route and vector badges must describe the measurements rather than infer magnitude from the recipe name.
      }
      if (item.name === 'unreachable' || item.name === 'future') {
        expect(trace.hasComputationalPath).toBe(false);
        expect(trace.nodes.some(node => node.route === 'local' && node.change !== 'exact')).toBe(true);
        expect(trace.nodes.filter(node => node.stage === 'logits' && node.position === item.predictionPosition).every(node => node.change === 'exact' && node.route === 'unaffected')).toBe(true);
      }
      expect(tf.memory().numTensors).toBe(count);
      expect(engine.cacheStats()).toEqual(cache);
      engine.releaseRun(comparison.modifiedRunId); engine.releaseRun(baseline.runId);
    } finally { engine.dispose(); }
  }
  expect(tf.memory().numTensors).toBe(initial);
});

it('validates pair ownership, supported interventions, comparison position, and released sessions', async () => {
  const engine = new RunEngine(verification.artifact), tokens = [0, 1, 5, 9];
  const patch: HeadScaleIntervention = { id: 'trace', kind: 'head.scale', layer: 0, head: 0, factor: .37, tokens: { kind: 'all' } };
  try {
    const baseline = await engine.runBaseline(tokens), other = await engine.runBaseline(tokens);
    const comparison = await engine.compare({ baselineRunId: baseline.runId, interventions: [patch], predictionPosition: 2 });
    const before = tf.memory().numTensors;
    await expect(engine.inspectEffectTrace(other.runId, comparison.modifiedRunId, 2)).rejects.toThrow('own baseline');
    await expect(engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, 3)).rejects.toThrow('completed comparison');
    await expect(engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, 2.5)).rejects.toThrow('completed comparison');
    expect(tf.memory().numTensors).toBe(before);
    const unsupported = await engine.compare({ baselineRunId: baseline.runId, interventions: [{ kind: 'residual.replace', id: 'a', layer: 0, stage: 'block.input', position: 3, values: Array(8).fill(0) }] });
    await expect(engine.inspectEffectTrace(baseline.runId, unsupported.modifiedRunId, 3)).rejects.toThrow('one head scaling');
    const multiple = await engine.compare({ baselineRunId: baseline.runId, interventions: [patch, { ...patch, id: 'second', head: 1 }] });
    await expect(engine.inspectEffectTrace(baseline.runId, multiple.modifiedRunId, 3)).rejects.toThrow('one head scaling');
    engine.releaseRun(comparison.modifiedRunId);
    await expect(engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, 2)).rejects.toThrow('unavailable');
  } finally { engine.dispose(); }
  await expect(engine.inspectEffectTrace('old', 'old', 0)).rejects.toThrow('ended');
});

it('cleans up a numerical failure and repeated traces', async () => {
  const initial = tf.memory().numTensors, engine = new RunEngine(verification.artifact);
  try {
    const baseline = await engine.runBaseline([0, 1, 5, 9]);
    const comparison = await engine.compare({ baselineRunId: baseline.runId, interventions: [{ id: 'test', kind: 'head.scale', layer: 0, head: 0, factor: .25, tokens: { kind: 'all' } }] });
    const count = tf.memory().numTensors;
    const bad = vi.spyOn(TransformerModel.prototype, 'readOut').mockImplementation(function(this: TransformerModel, rows) { return tf.fill([1, rows.shape[1], this.config.vocabularySize], NaN) as tf.Tensor3D; });
    try { await expect(engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, 3)).rejects.toThrow('non-finite'); }
    finally { bad.mockRestore(); }
    expect(tf.memory().numTensors).toBe(count);
    for (let i = 0; i < 15; i++) await engine.inspectEffectTrace(baseline.runId, comparison.modifiedRunId, 3);
    expect(tf.memory().numTensors).toBe(count);
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(initial);
});

it('keeps tiny nonzero differences, checks snapshot agreement, and bounds large captures', () => {
  expect(measureTrace([1, 2], [1, 2]).change).toBe('exact');
  expect(measureTrace([1, 2], [1 + 1e-8, 2]).change).toBe('small');
  expect(measureTrace([1, 2], [1, 3]).change).toBe('changed');
  expect(() => measureTrace([NaN], [1])).toThrow('non-finite');
  expect(() => checkTracePrediction([0, 1], [0, 1.1])).toThrow('completed prediction');
  expect(() => checkTracePrediction([0], [NaN])).toThrow('completed prediction');
  expect(traceBudget(retrieval.manifest.config, 32).captureBytes).toBeLessThan(TRACE_CAPTURE_BUDGET);
  expect(traceBudget({ ...retrieval.manifest.config, layers: 16, modelDim: 512, heads: 128, vocabularySize: 8192 }, 512).captureBytes).toBeGreaterThan(TRACE_CAPTURE_BUDGET);
});
