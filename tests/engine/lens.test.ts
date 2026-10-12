import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import type { HeadScaleIntervention } from '../../src/engine/contracts';
import fixture from '../fixtures/lens.json';
import { artifact, manifest } from './retrieval-artifact';
import { assertClose } from './numerical';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });

it('reads out baseline and modified residuals as the independent PyTorch logit lens does', async () => {
  expect(fixture.modelRevision).toBe(manifest.modelRevision);
  const start = tf.memory().numTensors;
  const engine = new RunEngine(artifact);
  const vocabulary = manifest.config.vocabularySize;
  let maximum = 0;
  try {
    const baseline = await engine.runBaseline(fixture.tokenIds);
    const compared = await engine.compare({ baselineRunId: baseline.runId, interventions: [fixture.intervention as HeadScaleIntervention], predictionPosition: fixture.predictionPosition });
    for (const [run, runId] of [['baseline', baseline.runId], ['modified', compared.modifiedRunId]] as const) {
      const lens = await engine.inspectResidualLens(runId, fixture.predictionPosition);
      expect(lens.stages).toEqual(fixture.stages);
      maximum = Math.max(maximum, assertClose(lens.logits, fixture[run].logits.flat(), `${run} lens logits`));
      assertClose(lens.norms, fixture[run].norms, `${run} residual norms`);
      // The read-out of the final residual is the model's own prediction. It is not bit-identical:
      // a 5-row matMul sums float32 products in a different order than the 16-position forward pass.
      const finalRow = lens.logits.slice((lens.stages.length - 1) * vocabulary);
      const logits = await engine.inspectActivation(runId, { stage: 'logits' }, { position: fixture.predictionPosition });
      assertClose(finalRow, logits.values, `${run} read-out versus forward logits`);
      assertClose(finalRow, fixture.finalLogits[run], `${run} final logits`);
    }
    await expect(engine.inspectResidualLens(baseline.runId, fixture.tokenIds.length)).rejects.toThrow('outside the input');
    await expect(engine.inspectResidualLens(baseline.runId, 1.5)).rejects.toThrow('outside the input');
    engine.releaseRun(compared.modifiedRunId); engine.releaseRun(baseline.runId);
    await expect(engine.inspectResidualLens(baseline.runId, 0)).rejects.toThrow('unavailable');
  } finally { engine.dispose(); }
  console.info('Maximum lens logit error', maximum);
  expect(tf.memory().numTensors).toBe(start);
});

it('returns embedding rows and token similarities as the independent PyTorch reference does', async () => {
  const { default: reference } = await import('../fixtures/embedding.json');
  expect(reference.modelRevision).toBe(manifest.modelRevision);
  const start = tf.memory().numTensors;
  const engine = new RunEngine(artifact);
  try {
    const baseline = await engine.runBaseline(reference.tokenIds);
    const compared = await engine.compare({ baselineRunId: baseline.runId, interventions: [fixture.intervention as HeadScaleIntervention], predictionPosition: fixture.predictionPosition });
    for (const expected of reference.positions) {
      const slice = await engine.inspectEmbedding(baseline.runId, expected.position);
      expect(slice.tokenId).toBe(expected.tokenId);
      assertClose(slice.token, expected.token, `token ${expected.position}`);
      assertClose(slice.positionEmbedding, expected.positionEmbedding, `position ${expected.position}`);
      assertClose(slice.sum, expected.sum, `sum ${expected.position}`);
      assertClose(slice.similarity, expected.similarity, `similarity ${expected.position}`);
      expect(slice.similarity[expected.tokenId]).toBeCloseTo(1, 6);
      // The model's own embedding output is the same sum, and head scaling cannot change it.
      const captured = await engine.inspectActivation(compared.modifiedRunId, { stage: 'embedding.output' }, { position: expected.position });
      expect(Array.from(captured.values)).toEqual(Array.from(slice.sum));
      const modified = await engine.inspectEmbedding(compared.modifiedRunId, expected.position);
      expect(Array.from(modified.sum)).toEqual(Array.from(slice.sum));
    }
    await expect(engine.inspectEmbedding(baseline.runId, -1)).rejects.toThrow('outside the input');
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(start);
});
