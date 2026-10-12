import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { canReachPrediction, type ModelConfig } from '../../src/engine/contracts';
import type { ExperimentRecipe } from '../../src/experiments';
import { compareLogits, defaultSettings, recipeExpectations, runCheck, summarize, summarizeChecks, validateSettings, type CombinationRecord } from '../../src/validation/report';

it('summarizes timings with the benchmark median and nearest-rank p95', () => {
  const values = Array.from({ length: 40 }, (_, i) => 40 - i);
  expect(summarize(values)).toEqual({ count: 40, medianMs: 20.5, p95Ms: 38, minimumMs: 1, maximumMs: 40 });
  expect(summarize([3])).toMatchObject({ medianMs: 3, p95Ms: 3 });
  expect(() => summarize([])).toThrow();
  expect(() => summarize([1, Number.NaN])).toThrow();
  expect(() => summarize([-1])).toThrow();
});

it('compares logits with the unchanged replay bound and rejects non-finite values', () => {
  expect(compareLogits([1, 100], [1, 100])).toEqual({ maximumError: 0, withinTolerance: true });
  expect(compareLogits([1 + 2.9e-5], [1]).withinTolerance).toBe(true);
  expect(compareLogits([1 + 3.1e-5], [1]).withinTolerance).toBe(false);
  expect(compareLogits([Number.NaN], [0])).toEqual({ maximumError: Infinity, withinTolerance: false });
  expect(() => compareLogits([1], [1, 2])).toThrow('length mismatch');
});

it('recipe expectations agree with the recorded PyTorch measurements and causal reachability', () => {
  const catalog = JSON.parse(readFileSync('public/models/retrieval/recipes.json', 'utf8')) as { recipes: ExperimentRecipe[] };
  const config = (JSON.parse(readFileSync('public/models/retrieval/manifest.json', 'utf8')) as { config: ModelConfig }).config;
  expect(catalog.recipes.map(recipe => recipe.id).sort()).toEqual(Object.keys(recipeExpectations).sort());
  for (const recipe of catalog.recipes) {
    const expectation = recipeExpectations[recipe.id];
    expect(expectation.holds(recipe.metrics), recipe.id).toBe(true);
    expect(canReachPrediction(recipe.intervention, recipe.predictionPosition, config, recipe.tokenIds.length), recipe.id).toBe(expectation.reachable);
  }
  // Each expectation distinguishes its recipe from the others' recorded behavior.
  expect(recipeExpectations.small.holds(catalog.recipes.find(recipe => recipe.id === 'effect')!.metrics)).toBe(false);
  expect(recipeExpectations.unreachable.holds(catalog.recipes.find(recipe => recipe.id === 'small')!.metrics)).toBe(false);
});

it('validates harness settings and keeps CPU as a required backend', () => {
  expect(validateSettings(defaultSettings)).toEqual(defaultSettings);
  expect(() => validateSettings({ ...defaultSettings, models: [] })).toThrow('checkpoint');
  expect(() => validateSettings({ ...defaultSettings, backends: [] })).toThrow('backend');
  expect(() => validateSettings({ ...defaultSettings, measuredSamples: 0 })).toThrow('Measured edits');
  expect(() => validateSettings({ ...defaultSettings, warmupSamples: 1.5 })).toThrow('Warm-up');
  const release = { name: 'release-identity', status: 'passed' as const };
  const combination = (backend: 'cpu' | 'webgl', status: 'passed' | 'unsupported' | 'failed'): CombinationRecord =>
    ({ model: 'retrieval', requestedBackend: backend, actualBackend: null, modelId: null, modelRevision: null, precision: null, checks: [{ name: 'worker-load', status }] });
  expect(summarizeChecks(release, [combination('cpu', 'passed'), combination('webgl', 'unsupported')], true)).toEqual({ passed: 2, failed: 0, unsupported: 1, skipped: 0, outcome: 'passed' });
  expect(summarizeChecks(release, [combination('cpu', 'unsupported')], true).outcome).toBe('failed');
  expect(summarizeChecks(release, [combination('webgl', 'failed')], true).outcome).toBe('failed');
  expect(summarizeChecks({ ...release, status: 'failed' }, [], true).outcome).toBe('failed');
  expect(summarizeChecks(release, [combination('cpu', 'passed')], false).outcome).toBe('incomplete');
});

it('records check failures without aborting the harness', async () => {
  await expect(runCheck('throws', async () => { throw new Error('broken'); })).resolves.toMatchObject({ name: 'throws', status: 'failed', detail: 'broken' });
  await expect(runCheck('reports', async () => ({ status: 'unsupported', detail: 'no context' }))).resolves.toMatchObject({ status: 'unsupported', detail: 'no context' });
  const passed = await runCheck('passes', async () => ({ data: { value: 1 } }));
  expect(passed).toMatchObject({ status: 'passed', data: { value: 1 } });
  expect(passed.durationMs).toBeGreaterThanOrEqual(0);
});
