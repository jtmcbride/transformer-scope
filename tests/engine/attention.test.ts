import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import { TransformerModel, disposeForward } from '../../src/engine/model';
import { canReachPrediction, normalizedAttention, type ActivationReference, type AttentionOverrideIntervention, type Intervention } from '../../src/engine/contracts';
import fixture from '../fixtures/attention-reference.json';
import { loadShippedArtifact } from './retrieval-artifact';
import { assertClose } from './numerical';

const verification = loadShippedArtifact('verification'), retrieval = loadShippedArtifact('retrieval');
const cases = fixture.cases as unknown as { name: string; tokenIds: number[]; interventions: Intervention[]; logits: number[][]; stages: Record<string, number[]> }[];
const T = 6, H = 2;
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });

it('full execution matches independent PyTorch attention overrides, alone and combined with other interventions', async () => {
  expect(fixture.verificationRevision).toBe(verification.manifest.modelRevision);
  const start = tf.memory().numTensors;
  const model = new TransformerModel(verification.manifest.config, verification.artifact.weights);
  let maximum = 0;
  try {
    const plain = model.forward(cases[0].tokenIds, [], [{ layer: 0, stage: 'attention.scores' }, { layer: 1, stage: 'attention.scores' }]);
    const baseline = { logits: Array.from(await plain.logits.data()), scores: [Array.from(await plain.captured['0:attention.scores'].data()), Array.from(await plain.captured['1:attention.scores'].data())] };
    disposeForward(plain);
    for (const item of cases) {
      const capture = Object.keys(item.stages).map(key => { const [layer, stage] = key.split(':'); return { layer: Number(layer), stage }; }) as ActivationReference[];
      const result = model.forward(item.tokenIds, item.interventions, capture);
      try {
        maximum = Math.max(maximum, assertClose(await result.logits.data(), item.logits.flat(), `${item.name} logits`));
        for (const [key, expected] of Object.entries(item.stages)) maximum = Math.max(maximum, assertClose(await result.captured[key].data(), expected, `${item.name} ${key}`));
        for (const patch of item.interventions) if (patch.kind === 'attention.override') {
          const offset = (patch.head * T + patch.query) * T;
          const row = Array.from((await result.captured[`${patch.layer}:attention.weights`].data()).slice(offset, offset + T));
          // The same float64 normalization on both sides gives identical float32 rows.
          expect(row).toEqual(Array.from(normalizedAttention(patch.weights)));
          expect(row).toEqual(Array.from(Float32Array.from(item.stages[`${patch.layer}:attention.weights`].slice(offset, offset + T))));
        }
        // Overrides act after the softmax: scores are untouched unless an earlier edit changed this layer's input.
        if (!item.interventions.some(patch => patch.kind === 'residual.replace'))
          for (const layer of [0, 1]) if (!item.interventions.some(patch => patch.kind === 'attention.override' && patch.layer < layer))
            expect(Array.from(await result.captured[`${layer}:attention.scores`].data())).toEqual(baseline.scores[layer]);
        if (item.name === 'final-layer-earlier-query') expect(Array.from((await result.logits.data()).slice(5 * 16))).toEqual(baseline.logits.slice(5 * 16));
      } finally { disposeForward(result); }
    }
  } finally { model.dispose(); }
  console.info('Maximum attention-override error', maximum);
  expect(tf.memory().numTensors).toBe(start);
});

it('plans full execution, reports reachability, and inspects overridden runs without the cache', async () => {
  const start = tf.memory().numTensors;
  const engine = new RunEngine(verification.artifact);
  try {
    for (const item of cases) {
      const baseline = await engine.runBaseline(item.tokenIds);
      const position = item.tokenIds.length - 1;
      const result = await engine.compare({ baselineRunId: baseline.runId, interventions: item.interventions, predictionPosition: position });
      expect(result.execution.strategy, item.name).toBe('full');
      expect(result.hasComputationalPath, item.name).toBe(item.interventions.some(patch => canReachPrediction(patch, position, verification.manifest.config, T)));
      assertClose((await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position })).values, item.logits[position], `${item.name} engine logits`);
      for (const layer of [0, 1]) {
        const weights = await engine.inspectActivation(result.modifiedRunId, { layer, stage: 'attention.weights' });
        assertClose(weights.values, item.stages[`${layer}:attention.weights`], `${item.name} engine weights ${layer}`);
      }
      engine.releaseRun(baseline.runId); engine.releaseRun(result.modifiedRunId);
    }
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(start);
});

it('rejects invalid overrides before allocating tensors', async () => {
  const engine = new RunEngine(verification.artifact);
  const baseline = await engine.runBaseline([0, 1, 5, 9]);
  const count = tf.memory().numTensors;
  const valid: AttentionOverrideIntervention = { id: 'a', kind: 'attention.override', layer: 0, head: 1, query: 2, weights: [1, 1, 0, 0] };
  const invalid: [Partial<AttentionOverrideIntervention>, string][] = [
    [{ weights: [1, 1, 0] }, 'one value per input position (4)'], [{ weights: [1, 0, 0, 1] }, 'later position'],
    [{ weights: [1, -1, 1, 0] }, 'non-negative'], [{ weights: [1, Number.NaN, 0, 0] }, 'finite'], [{ weights: [0, 0, 0, 0] }, 'positive sum'],
    [{ query: 4 }, 'outside the input'], [{ head: 2 }, 'outside the model'], [{ layer: -1 }, 'outside the model'],
  ];
  try {
    for (const [patch, message] of invalid)
      await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [{ ...valid, ...patch } as AttentionOverrideIntervention] })).rejects.toThrow(message);
    await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [valid, { ...valid, id: 'b' }] })).rejects.toThrow('only once');
    expect(tf.memory().numTensors).toBe(count);
    expect(Array.from(normalizedAttention([1, 1, 2, 0]))).toEqual([.25, .25, .5, 0]);
  } finally { engine.dispose(); }
});

it('reproduces the independent retrieval routing sweep', async () => {
  const { routing } = fixture;
  expect(fixture.retrievalRevision).toBe(retrieval.manifest.modelRevision);
  const engine = new RunEngine(retrieval.artifact);
  let maximum = 0;
  try {
    const baseline = await engine.runBaseline(routing.tokenIds);
    const rows = await engine.inspectActivation(baseline.runId, { layer: 1, stage: 'attention.weights' }, { queryPosition: 15 });
    for (const [head, row] of Object.entries(routing.baselineRows)) assertClose(rows.values.slice(Number(head) * 16, Number(head) * 16 + 16), row, `baseline row ${head}`);
    for (const entry of routing.sweep) {
      const weights = Array.from({ length: 16 }, (_, i) => i === entry.position ? 1 : 0);
      const result = await engine.compare({ baselineRunId: baseline.runId, predictionPosition: 15,
        interventions: [{ id: 'route', kind: 'attention.override', layer: entry.layer, head: entry.head, query: entry.query, weights }] });
      const logits = await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position: 15 });
      maximum = Math.max(maximum, assertClose(logits.values, entry.logits, `L${entry.layer + 1}H${entry.head + 1}→${entry.position}`));
      engine.releaseRun(result.modifiedRunId);
    }
  } finally { engine.dispose(); }
  console.info('Maximum routing-sweep logit error', maximum);
});
