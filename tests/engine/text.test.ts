import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { RunEngine } from '../../src/engine/runs';
import { tokenize, textForTokens, tokenLabel, type HeadScaleIntervention } from '../../src/engine/contracts';
import { createExperiment, validateExperiment } from '../../src/engine/experiment';
import { loadShippedArtifact } from './retrieval-artifact';
import fixture from '../fixtures/text-inference.json';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
function parity(actual: ArrayLike<number>, expected: number[]) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, i) => expect(Math.abs(actual[i] - value), `logit ${i}`).toBeLessThanOrEqual(2e-5 + 1e-5 * Math.abs(value)));
}
it('the shipped TinyStories checkpoint matches independent Python text logits and full/suffix interventions', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(fixture.modelRevision);
  expect(loaded.manifest.weightsSha256).toBe(fixture.weightsSha256);
  const engine = new RunEngine(loaded.artifact);
  try {
    for (const example of fixture.cases) {
      const ids = tokenize(example.text, loaded.tokenizer, 64);
      expect(ids).toEqual(example.tokenIds);
      expect(textForTokens(ids, loaded.tokenizer)).toBe(example.text);
      const baseline = await engine.runBaseline(ids);
      parity((await engine.inspectActivation(baseline.runId, { stage: 'logits' }, { position: ids.length - 1 })).values, example.logits);
      for (const variant of example.variants) {
        const patch = variant.intervention as HeadScaleIntervention;
        const cached = await engine.compare({ baselineRunId: baseline.runId, interventions: [patch], predictionPosition: ids.length - 1, targetTokenId: 259 });
        const full = await engine.compare({ baselineRunId: baseline.runId, interventions: [patch], predictionPosition: ids.length - 1, targetTokenId: 259, executionPreference: 'full' });
        parity((await engine.inspectActivation(cached.modifiedRunId, { stage: 'logits' }, { position: ids.length - 1 })).values, variant.logits);
        parity((await engine.inspectActivation(full.modifiedRunId, { stage: 'logits' }, { position: ids.length - 1 })).values, variant.logits);
        engine.releaseRun(cached.modifiedRunId); engine.releaseRun(full.modifiedRunId);
      }
      engine.releaseRun(baseline.runId);
    }
  } finally { engine.dispose(); }
}, 60_000);

it('text export/replay preserves UTF-8 and whitespace, and display labels cannot become encoded text', () => {
  const { tokenizer, manifest } = loadShippedArtifact('tinystories');
  const example = fixture.cases[2];
  const ids = tokenize(example.text, tokenizer, 64);
  const document = createExperiment(manifest, tokenizer, ids,
    { predictionPosition: ids.length - 1, targetTokenId: 10, interventions: [example.variants[0].intervention as HeadScaleIntervention] },
    example.logits, example.variants[0].logits, { backend: 'cpu', precision: { dtype: 'float32', float32Textures: null, webglPacking: null } });
  expect(document.input.text).toBe(example.text);
  validateExperiment(document, manifest, tokenizer);
  expect(ids.some(id => tokenLabel(tokenizer, id).includes('␠'))).toBe(true);
  const changed = structuredClone(document); changed.input.text = changed.input.text.replace('\n', ' ');
  expect(() => validateExperiment(changed, manifest, tokenizer)).toThrow('token IDs do not agree');
  expect(() => tokenize('Hello '.repeat(80), tokenizer, 64)).toThrow('shorten');
});
