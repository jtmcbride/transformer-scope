import { readFileSync } from 'node:fs';
import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import { runProbe, type ProbeFixture } from '../../src/engine/probe';
import { validateExperiments, retrievalAnswer } from '../../src/experiments';
import { assertClose } from './numerical';
import { artifact, json, manifest, tokenizer } from './retrieval-artifact';

const data = [json('recipes.json'), json('evaluation.json')];
const metadata = validateExperiments(data, manifest, tokenizer);

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });

it('reproduces trained PyTorch forward, every gradient, and Adam without retaining tensors', async () => {
  const report = await runProbe(artifact, json('probe.json') as ProbeFixture);
  console.info('Trained CPU probe', JSON.stringify(report));
  expect(report.retainedTensorDelta).toBe(0);
});

it('reproduces all trained recipes with cached and full execution', async () => {
  const start = tf.memory().numTensors;
  const engine = new RunEngine(artifact);
  let maximum = 0;
  try {
    for (const recipe of metadata.recipes) {
      expect(retrievalAnswer(recipe.tokenIds, tokenizer)).toBe(recipe.targetTokenId);
      const baseline = await engine.runBaseline(recipe.tokenIds);
      const baseSlice = await engine.inspectActivation(baseline.runId, { stage: 'logits' }, { position: recipe.predictionPosition });
      maximum = Math.max(maximum, assertClose(baseSlice.values, recipe.metrics.baselineLogits, `${recipe.id}/baseline`));
      const request = { baselineRunId: baseline.runId, interventions: [recipe.intervention], predictionPosition: recipe.predictionPosition, targetTokenId: recipe.targetTokenId };
      const cached = await engine.compare(request);
      const full = await engine.compare({ ...request, executionPreference: 'full' });
      expect(cached.execution.strategy).toBe('cached-suffix');
      expect(cached.hasComputationalPath).toBe(recipe.id !== 'unreachable');
      for (const result of [cached, full]) {
        const logits = await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position: recipe.predictionPosition });
        maximum = Math.max(maximum, assertClose(logits.values, recipe.metrics.modifiedLogits, `${recipe.id}/${result.execution.strategy}`));
        expect(Math.abs(result.metrics.targetLogProbabilityDelta! - recipe.metrics.targetLogProbabilityDelta)).toBeLessThan(2e-5);
        expect(Math.abs(result.metrics.jsDivergence - recipe.metrics.jsDivergence)).toBeLessThan(2e-6);
        engine.releaseRun(result.modifiedRunId);
      }
      if (recipe.id === 'effect') expect(cached.metrics.targetLogProbabilityDelta).toBeLessThan(-1);
      if (recipe.id === 'small') expect(cached.metrics.jsDivergence).toBeLessThan(.001);
      if (recipe.id === 'unreachable') expect(cached.metrics.maxAbsoluteLogitChange).toBe(0);
      engine.releaseRun(baseline.runId);
    }
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(start);
  console.info(`Trained recipe maximum absolute logit error: ${maximum}`);
});

it('rejects stale recipes and malformed inputs and clears ambiguous answer inference', () => {
  const stale = structuredClone(data); stale[0].modelRevision = '0'.repeat(64);
  expect(() => validateExperiments(stale, manifest, tokenizer)).toThrow(/match/);
  const bad = structuredClone(data); bad[0].recipes[0].text = 'A = 1';
  expect(() => validateExperiments(bad, manifest, tokenizer)).toThrow(/input/);
  const duplicates = structuredClone(data); duplicates[0].recipes[1].id = duplicates[0].recipes[0].id;
  expect(() => validateExperiments(duplicates, manifest, tokenizer)).toThrow(/Invalid/);
  expect(retrievalAnswer([0, 1, 9, 5, 10, 1, 9, 6, 10, 11, 1, 9], tokenizer)).toBeUndefined();
});

it('matches independent logits and answers for all 453 held-out sequences', async () => {
  const fixture = JSON.parse(readFileSync(new URL('../fixtures/retrieval-evaluation.json', import.meta.url), 'utf8')) as {
    modelRevision: string; cases: { tokenIds: number[]; target: number; logits: number[] }[] };
  expect(fixture.modelRevision).toBe(manifest.modelRevision);
  expect(fixture.cases).toHaveLength(metadata.evaluation.heldOut.count);
  const engine = new RunEngine(artifact, { cacheBudgetBytes: 0 });
  let correct = 0, maximum = 0;
  try {
    for (const item of fixture.cases) {
      const baseline = await engine.runBaseline(item.tokenIds);
      const logits = await engine.inspectActivation(baseline.runId, { stage: 'logits' }, { position: item.tokenIds.length - 1 });
      maximum = Math.max(maximum, assertClose(logits.values, item.logits, 'held-out logits'));
      if (logits.values.indexOf(Math.max(...logits.values)) === item.target) correct++;
      engine.releaseRun(baseline.runId);
    }
  } finally { engine.dispose(); }
  expect(correct / fixture.cases.length).toBe(metadata.evaluation.heldOut.accuracy);
  console.info(`Held-out CPU accuracy ${correct}/${fixture.cases.length}, maximum logit error ${maximum}`);
}, 30_000);
