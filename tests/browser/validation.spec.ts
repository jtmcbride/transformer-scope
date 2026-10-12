import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import type { CheckRecord, ValidationReport } from '../../src/validation/report';

async function configure(page: Page, options: { models: string[]; backends: string[]; warmup: number; measured: number }) {
  await page.getByLabel('Evidence class').selectOption('automation');
  await page.getByLabel('Device description').fill(`Playwright ${test.info().project.name}; not a physical device`);
  for (const [label, value] of [['Trained retrieval checkpoint', 'retrieval'], ['Verification checkpoint', 'verification']] as const)
    await page.getByLabel(label).setChecked(options.models.includes(value));
  for (const [label, value] of [['CPU worker', 'cpu'], ['WebGL worker', 'webgl']] as const)
    await page.getByLabel(label).setChecked(options.backends.includes(value));
  await page.getByLabel('Warm-up edits').fill(String(options.warmup));
  await page.getByLabel('Measured edits').fill(String(options.measured));
}
async function finish(page: Page): Promise<ValidationReport> {
  await expect(page.getByTestId('validation-outcome')).toBeVisible({ timeout: 180_000 });
  return JSON.parse(await page.getByTestId('validation-json').textContent() ?? '') as ValidationReport;
}
const check = (report: ValidationReport, model: string, backend: string, name: string): CheckRecord =>
  report.combinations.find(item => item.model === model && item.requestedBackend === backend)!.checks.find(item => item.name === name)!;

test('the validation harness is opt-in and records identity, numerical checks, and separated timings', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  const requests: string[] = [], errors: string[] = [];
  page.on('request', request => requests.push(request.url()));
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible({ timeout: 30_000 });
  expect(requests.filter(url => /validation/u.test(url))).toEqual([]);
  requests.length = 0;

  await page.goto('validation.html');
  await expect(page.getByRole('button', { name: 'Start validation' })).toBeEnabled();
  await page.waitForTimeout(500);
  expect(requests.filter(url => /model\.worker|\/models\/|release\.json/u.test(url))).toEqual([]);
  await expect(page.locator('iframe')).toHaveCount(0);

  await configure(page, { models: ['retrieval', 'verification'], backends: ['cpu', 'webgl'], warmup: 1, measured: 3 });
  await page.getByRole('button', { name: 'Start validation' }).click();
  const report = await finish(page);
  await testInfo.attach('device-validation-report', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });

  expect(report.format).toBe('transformer-microscope-device-validation');
  expect(report.evidence).toMatchObject({ evidenceClass: 'automation', automated: true, pageHiddenDuringRun: false });
  expect(report.release.status).toBe('passed');
  expect((report.release.data as { commit: string }).commit).toMatch(/^[0-9a-f]{40}$/u);
  expect(report.environment.userAgent).toBe(await page.evaluate(() => navigator.userAgent));
  expect(report.combinations).toHaveLength(4);
  for (const combination of report.combinations) {
    if (combination.requestedBackend === 'webgl' && check(report, combination.model, 'webgl', 'worker-load').status === 'unsupported') {
      testInfo.annotations.push({ type: 'unsupported-webgl', description: check(report, combination.model, 'webgl', 'worker-load').detail ?? '' });
      expect(combination.checks.filter(item => item.status === 'failed')).toEqual([]);
      continue;
    }
    expect(combination.actualBackend).toBe(combination.requestedBackend);
    expect(combination.modelRevision).toBe((report.release.data as { models: Record<string, { modelRevision: string }> }).models[combination.model].modelRevision);
    const expected = ['ui-cold-readiness', 'ui-warm-interaction', 'ui-baseline-reset', 'ui-cpu-recovery', 'worker-load', 'computation-rpc', 'reference-probe', 'prepared-recipes', 'replay-roundtrip', 'resource-cleanup'];
    expect(combination.checks.map(item => item.name)).toEqual(expected);
    for (const item of combination.checks)
      expect(item.status, `${combination.model}/${combination.requestedBackend}/${item.name}: ${item.detail}`).toBe(combination.model === 'verification' && item.name === 'prepared-recipes' ? 'skipped' : 'passed');
    const ui = check(report, combination.model, combination.requestedBackend, 'ui-warm-interaction').data as { layers: { domCommit: { count: number; p95Ms: number }; frameOpportunity: { count: number; p95Ms: number } }[] };
    expect(ui.layers).toHaveLength(2);
    for (const layer of ui.layers) {
      expect(layer.domCommit.count).toBe(3);
      expect(layer.frameOpportunity.count).toBe(3);
      expect(layer.frameOpportunity.p95Ms).toBeGreaterThanOrEqual(layer.domCommit.p95Ms);
    }
    const rpc = check(report, combination.model, combination.requestedBackend, 'computation-rpc').data as { layers: { fullRoundTrip: { count: number }; cachedWorkerComputation: { count: number } }[] };
    expect(rpc.layers.map(layer => [layer.fullRoundTrip.count, layer.cachedWorkerComputation.count])).toEqual([[3, 3], [3, 3]]);
    expect((check(report, combination.model, combination.requestedBackend, 'ui-cold-readiness').data as { coldReadyMs: number }).coldReadyMs).toBeGreaterThan(0);
  }
  const recipes = check(report, 'retrieval', 'cpu', 'prepared-recipes').data as { cases: { id: string; hasComputationalPath: boolean; cachedMaximumError: number }[] };
  expect(recipes.cases.map(item => [item.id, item.hasComputationalPath])).toEqual([['effect', true], ['small', true], ['unreachable', false]]);
  expect(report.summary.outcome).toBe('passed');
  await expect(page.locator('iframe')).toHaveCount(0);

  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download JSON' }).click();
  expect(JSON.parse(await readFile((await (await download).path())!, 'utf8'))).toEqual(report);
  expect(errors).toEqual([]);
});

test('the validation harness reports numerical failures and unsupported WebGL explicitly', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  // Emulate a worker without WebGL, as in capability.spec.ts, and corrupt one reference value.
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `if (typeof OffscreenCanvas !== 'undefined') {
      const originalContext = OffscreenCanvas.prototype.getContext;
      OffscreenCanvas.prototype.getContext = function(kind, ...args) { return kind.includes('webgl') ? null : originalContext.call(this, kind, ...args); };
    }\n${await response.text()}` });
  });
  await page.route('**/models/retrieval/probe.json', async route => {
    const response = await route.fetch();
    const probe = await response.json();
    const first = Object.values(probe.cases[0].activations)[0] as { values: number[] };
    first.values[0] += 1;
    await route.fulfill({ response, json: probe });
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('validation.html');
  await configure(page, { models: ['retrieval'], backends: ['cpu', 'webgl'], warmup: 0, measured: 2 });
  await page.getByRole('button', { name: 'Start validation' }).click();
  const report = await finish(page);
  await testInfo.attach('device-validation-failure-report', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
  expect(report.summary.outcome).toBe('failed');
  expect(check(report, 'retrieval', 'cpu', 'reference-probe')).toMatchObject({ status: 'failed' });
  expect(check(report, 'retrieval', 'cpu', 'reference-probe').detail).toContain('Probe mismatch');
  for (const name of ['ui-cold-readiness', 'ui-warm-interaction', 'ui-cpu-recovery', 'computation-rpc', 'prepared-recipes', 'replay-roundtrip', 'resource-cleanup'])
    expect(check(report, 'retrieval', 'cpu', name).status, name).toBe('passed');
  expect(check(report, 'retrieval', 'webgl', 'worker-load')).toMatchObject({ status: 'unsupported' });
  expect(check(report, 'retrieval', 'webgl', 'ui-cold-readiness').status).toBe('unsupported');
  expect(check(report, 'retrieval', 'webgl', 'reference-probe').status).toBe('skipped');
  await expect(page.locator('tr.failed')).toHaveCount(1);
  await expect(page.locator('tr.unsupported')).toHaveCount(2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
