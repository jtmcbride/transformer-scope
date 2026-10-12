import { expect, it } from 'vitest';
import reference from '../fixtures/reference.json';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import { createExperiment, parseExperiment, validateExperiment, checkReplay, MAX_EXPERIMENT_BYTES } from '../../src/engine/experiment';
import { comparePredictions } from '../../src/engine/comparison';
import type { HeadScaleIntervention } from '../../src/engine/contracts';

const manifest = validateManifest(reference.manifest);
const tokenizer = validateTokenizer({ schemaVersion: 1, kind: 'symbols', bosTokenId: 0,
  tokens: ['<BOS>', ...'ABCDEFGHIJKLMNO'] }, manifest.config.vocabularySize);
const original = reference.cases.find(item => item.name === 'length-5-case-0')!;
const modified = reference.cases.find(item => item.name === 'length-5-case-2')!;
const snapshot = () => createExperiment(manifest, tokenizer, [0, ...original.tokenIds],
  { predictionPosition: 2, targetTokenId: 13, interventions: modified.interventions as HeadScaleIntervention[] },
  original.activations.logits.values.slice(32, 48), modified.activations.logits.values.slice(32, 48),
  { backend: 'cpu', precision: { dtype: 'float32', float32Textures: null, webglPacking: null } });

it('round-trips complete distributions, nonfinal positions, exact input and checkpoint provenance', () => {
  const document = snapshot();
  const parsed = parseExperiment(JSON.stringify(document));
  expect(parsed).toEqual(document);
  expect(parsed.input.text).toBe('A E I C G');
  expect(parsed.observation.predictionPosition).toBe(2);
  expect(parsed.results.baselineLogits).toHaveLength(16);
  validateExperiment(parsed, manifest, tokenizer);
  const replay = checkReplay(parsed, parsed.results.baselineLogits, parsed.results.modifiedLogits, 'webgl');
  expect(replay).toEqual({ matches: true, baselineMaximumError: 0, modifiedMaximumError: 0, savedBackend: 'cpu', currentBackend: 'webgl' });
  parsed.input.tokenIds[0] = 7;
  expect(document.input.tokenIds[0]).toBe(0);
});

it('rejects malformed, oversized, incompatible, ambiguous and tampered experiment files', () => {
  expect(() => parseExperiment('oops')).toThrow(/JSON/);
  expect(() => parseExperiment(' '.repeat(MAX_EXPERIMENT_BYTES + 1))).toThrow(/1 MiB/);
  expect(() => parseExperiment(JSON.stringify({ ...snapshot(), schemaVersion: 2 }))).toThrow(/version 1/);
  const bad = snapshot(); bad.checkpoint.weightsSha256 = '0'.repeat(64);
  expect(() => validateExperiment(bad, manifest, tokenizer)).toThrow(/checkpoint/);
  const text = snapshot(); text.input.text = 'A A';
  expect(() => validateExperiment(text, manifest, tokenizer)).toThrow(/token IDs/);
  const metric = snapshot(); metric.results.metrics.jsDivergence += .1;
  expect(() => validateExperiment(metric, manifest, tokenizer)).toThrow(/metrics/);
  const selection = snapshot(); selection.observation.interventions[0].tokens = { kind: 'indices', indices: [0, 1] };
  expect(() => validateExperiment(selection, manifest, tokenizer)).toThrow(/single-position/);
  const invalid = snapshot(); invalid.observation.predictionPosition = invalid.input.tokenIds.length;
  expect(() => validateExperiment(invalid, manifest, tokenizer)).toThrow(/position/);
});

it('reports finite replay discrepancies separately from valid saved-file consistency', () => {
  const document = snapshot();
  const altered = document.results.modifiedLogits.map(value => value + .1);
  const report = checkReplay(document, document.results.baselineLogits, altered, 'cpu');
  expect(report.matches).toBe(false);
  expect(report.modifiedMaximumError).toBeCloseTo(.1, 12);
  // A self-consistent but stale measurement is importable; replay must expose the discrepancy.
  const saved = snapshot(); saved.results.modifiedLogits = altered;
  saved.results.metrics = comparePredictions(saved.results.baselineLogits, altered, tokenizer, 8, saved.observation.targetTokenId).metrics;
  validateExperiment(saved, manifest, tokenizer);
  expect(checkReplay(saved, document.results.baselineLogits, document.results.modifiedLogits, 'cpu').matches).toBe(false);
});
