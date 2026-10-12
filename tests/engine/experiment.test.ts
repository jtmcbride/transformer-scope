import { expect, it } from 'vitest';
import reference from '../fixtures/reference.json';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import { createExperiment, parseExperiment, validateExperiment, checkReplay, MAX_EXPERIMENT_BYTES } from '../../src/engine/experiment';
import { comparePredictions } from '../../src/engine/comparison';
import type { AttentionOverrideIntervention, HeadScaleIntervention, Intervention, ResidualReplaceIntervention } from '../../src/engine/contracts';

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
  expect(() => parseExperiment(JSON.stringify({ ...snapshot(), schemaVersion: 3 }))).toThrow(/version 1 or 2/);
  const bad = snapshot(); bad.checkpoint.weightsSha256 = '0'.repeat(64);
  expect(() => validateExperiment(bad, manifest, tokenizer)).toThrow(/checkpoint/);
  const text = snapshot(); text.input.text = 'A A';
  expect(() => validateExperiment(text, manifest, tokenizer)).toThrow(/token IDs/);
  const metric = snapshot(); metric.results.metrics.jsDivergence += .1;
  expect(() => validateExperiment(metric, manifest, tokenizer)).toThrow(/metrics/);
  const selection = snapshot(); (selection.observation.interventions[0] as HeadScaleIntervention).tokens = { kind: 'indices', indices: [0, 1] };
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

it('writes version 2 and still imports version 1 head-scaling files', () => {
  const current = snapshot();
  expect(current.schemaVersion).toBe(2);
  const legacy = { ...snapshot(), schemaVersion: 1 as const };
  validateExperiment(parseExperiment(JSON.stringify(legacy)), manifest, tokenizer);
  const redirected = { ...legacy, observation: { ...legacy.observation,
    interventions: [{ id: 'a', kind: 'attention.override', layer: 0, head: 0, query: 2, weights: [1, 0, 0, 0, 0, 0] }] as Intervention[] } };
  expect(() => parseExperiment(JSON.stringify(redirected))).toThrow(/one saved head-scaling/);
});

const logits = (offset: number) => original.activations.logits.values.slice(32 + offset, 48 + offset);
const save = (interventions: Intervention[], sourceTokenIds?: number[]) => createExperiment(manifest, tokenizer, [0, ...original.tokenIds],
  { predictionPosition: 4, targetTokenId: 3, interventions, sourceTokenIds }, logits(0), logits(16),
  { backend: 'cpu', precision: { dtype: 'float32', float32Textures: null, webglPacking: null } });

it('round-trips one-hot and even attention redirects, and rejects rows the screen cannot restore', () => {
  for (const weights of [[0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]) {
    const document = save([{ id: 'a', kind: 'attention.override', layer: 1, head: 1, query: 4, weights }]);
    expect(parseExperiment(JSON.stringify(document))).toEqual(document);
  }
  const partial = save([{ id: 'a', kind: 'attention.override', layer: 1, head: 1, query: 4, weights: [1, 0, 0, 0, 0, 0] }]);
  (partial.observation.interventions[0] as AttentionOverrideIntervention).weights = [.5, .5, 0, 0, 0, 0];
  expect(() => parseExperiment(JSON.stringify(partial))).toThrow(/one-position or even/);
  (partial.observation.interventions[0] as AttentionOverrideIntervention).weights = [1, 1, 0, 1, 0, 0];
  expect(() => parseExperiment(JSON.stringify(partial))).toThrow(/one-position or even/);
});

it('round-trips a residual patch with its source input and rejects missing or mismatched sources', () => {
  const source = [0, 1, 5, 9, 3, 8];
  const patch: ResidualReplaceIntervention = { id: 'p', kind: 'residual.replace', layer: 0, stage: 'attention.residual', position: 5, values: Array(8).fill(.25) };
  const document = save([patch], source);
  expect(document.observation.source).toEqual({ text: 'A E I C H', tokenIds: source });
  const parsed = parseExperiment(JSON.stringify(document));
  expect(parsed).toEqual(document);
  validateExperiment(parsed, manifest, tokenizer);
  const missing = structuredClone(document); delete missing.observation.source;
  expect(() => parseExperiment(JSON.stringify(missing))).toThrow(/source input/);
  const extra = save([{ id: 'h', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'all' } }]);
  extra.observation.source = document.observation.source;
  expect(() => parseExperiment(JSON.stringify(extra))).toThrow(/Only a saved residual patch/);
  const short = structuredClone(document); short.observation.source = { text: 'A E', tokenIds: [0, 1, 5] };
  expect(() => validateExperiment(short, manifest, tokenizer)).toThrow(/same length/);
  const mismatched = structuredClone(document); mismatched.observation.source!.text = 'A E I C G';
  expect(() => validateExperiment(mismatched, manifest, tokenizer)).toThrow(/source text/);
  const wrongWidth = structuredClone(document); (wrongWidth.observation.interventions[0] as ResidualReplaceIntervention).values = [1, 2];
  expect(() => validateExperiment(wrongWidth, manifest, tokenizer)).toThrow(/8 finite values/);
  const two = structuredClone(document); two.observation.interventions.push(patch);
  expect(() => parseExperiment(JSON.stringify(two))).toThrow(/exactly one intervention/);
});
