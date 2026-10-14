import { expect, test } from '@playwright/test';
import type { Commands, Command, WorkerResponse } from '../../src/workers/protocol';
import type { ExperimentRecipe } from '../../src/experiments';

test('trained checkpoint reproduces reference probes and prepared recipes on CPU and WebGL', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  let workerUrl = '';
  page.on('request', request => { if (/model\.worker-.*\.js/u.test(request.url())) workerUrl = request.url(); });
  await page.goto('?view=details&model=retrieval');
  await expect(page.getByTestId('recipe-note')).toContainText('Ablation changes the answer', { timeout: 30_000 });
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible();
  const report = await page.evaluate(async url => {
    const worker = new Worker(url, { type: 'module' });
    let id = 0;
    const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data, call = pending.get(response.id);
      if (!call) return;
      pending.delete(response.id);
      if (response.ok) call.resolve(response.result);
      else call.reject(new Error(`${response.error.code}: ${response.error.message}`));
    };
    const request = <K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> =>
      new Promise((resolve, reject) => {
        const current = ++id;
        pending.set(current, { resolve: value => resolve(value as Commands[K]['response']), reject });
        worker.postMessage({ id: current, command, payload });
      });
    const manifestUrl = new URL('models/retrieval/manifest.json', window.location.href).href;
    const catalog = await (await fetch(new URL('recipes.json', manifestUrl))).json() as { recipes: ExperimentRecipe[] };
    function close(a: ArrayLike<number>, b: ArrayLike<number>, label: string) {
      if (a.length !== b.length) throw new Error(`Shape mismatch: ${label}`);
      let maximum = 0;
      for (let i = 0; i < a.length; i++) {
        const error = Math.abs(a[i] - b[i]);
        if (!Number.isFinite(a[i]) || error > 2e-5 + 1e-5 * Math.abs(b[i])) throw new Error(`${label}[${i}]: ${error}`);
        maximum = Math.max(maximum, error);
      }
      return maximum;
    }
    const reports = [];
    try {
      for (const backend of ['cpu', 'webgl'] as const) {
        let loaded: Commands['LOAD']['response'];
        try { loaded = await request('LOAD', { backend, manifestUrl }); }
        catch (error) {
          if (backend !== 'webgl' || !(error instanceof Error) || !error.message.startsWith('UNSUPPORTED_BACKEND')) throw error;
          reports.push({ backend, status: 'unsupported' as const, reason: error.message }); continue;
        }
        const probe = await request('PROBE', {});
        const cases = [];
        for (const recipe of catalog.recipes) {
          const baseline = await request('BASELINE', { tokenIds: recipe.tokenIds });
          const baseLogits = await request('INSPECT', { runId: baseline.runId, reference: { stage: 'logits' }, selection: { position: recipe.predictionPosition } });
          let maximum = close(baseLogits.values, recipe.metrics.baselineLogits, `${recipe.id}/baseline`);
          const payload = { baselineRunId: baseline.runId, interventions: [recipe.intervention], predictionPosition: recipe.predictionPosition, targetTokenId: recipe.targetTokenId };
          const cached = await request('COMPARE', payload);
          const full = await request('COMPARE', { ...payload, executionPreference: 'full' });
          for (const result of [cached, full]) {
            const logits = await request('INSPECT', { runId: result.modifiedRunId, reference: { stage: 'logits' }, selection: { position: recipe.predictionPosition } });
            maximum = Math.max(maximum, close(logits.values, recipe.metrics.modifiedLogits, `${recipe.id}/${result.execution.strategy}`));
            await request('RELEASE', { runId: result.modifiedRunId });
          }
          cases.push({ id: recipe.id, maximumLogitError: maximum, cachedMetrics: cached.metrics, fullMetrics: full.metrics,
            hasComputationalPath: cached.hasComputationalPath, strategy: cached.execution.strategy });
          await request('RELEASE', { runId: baseline.runId });
        }
        const resources = await request('RESOURCES', {});
        reports.push({ backend: loaded.backend, status: 'passed' as const, probe, cases, resources });
      }
      return { userAgent: navigator.userAgent, reports };
    } finally { worker.terminate(); }
  }, workerUrl);
  expect(report.reports[0].status).toBe('passed');
  for (const result of report.reports) {
    if (result.status === 'unsupported') { testInfo.annotations.push({ type: 'unsupported-webgl', description: result.reason }); continue; }
    expect(result.probe.retainedTensorDelta).toBe(0);
    expect(result.resources.tensorCount).toBe(36);
    expect(result.resources.cache.logicalBytes).toBe(0);
    for (const recipe of result.cases) {
      expect(recipe.strategy).toBe('cached-suffix');
      expect(recipe.hasComputationalPath).toBe(recipe.id !== 'unreachable');
      expect(recipe.maximumLogitError).toBeLessThan(2e-5);
      if (recipe.id === 'effect') expect(recipe.cachedMetrics.targetLogProbabilityDelta).toBeLessThan(-1);
      if (recipe.id === 'small') expect(recipe.cachedMetrics.jsDivergence).toBeLessThan(.001);
      if (recipe.id === 'unreachable') expect(recipe.cachedMetrics.maxAbsoluteLogitChange).toBe(0);
    }
  }
  await testInfo.attach('trained-worker-results', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
});

test('prepared experiments, custom inputs, checkpoint switching, and mobile matrix stay coherent', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('?view=details&model=retrieval');
  await expect(page.getByTestId('recipe-note')).toContainText('Ablation changes the answer', { timeout: 30_000 });
  await expect(page.getByText('100.0% held-out answer accuracy')).toBeVisible();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible();
  await page.getByLabel('Prepared experiment').selectOption('unreachable');
  await expect(page.getByText('This setting has no active computational path', { exact: false })).toBeVisible();
  await expect(page.getByLabel('Layer', { exact: true })).toHaveValue('1');
  await expect(page.getByLabel('Apply at')).toHaveValue('0');
  await page.getByLabel('Prepared experiment').selectOption('small');
  await expect(page.getByTestId('recipe-note')).toContainText('A reachable edit has little effect');
  await expect(page.getByText('This setting has no active computational path', { exact: false })).toHaveCount(0);
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.getByTestId('recipe-note')).toHaveCount(0);
  await expect(page.getByLabel('Prepared experiment')).toHaveValue('');
  await page.getByRole('button', { name: 'Edit tokens', exact: true }).click();
  await page.getByLabel('Space-separated symbols').fill('B = 3 ; D = 0 ; C = 1 ; ? B =');
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.getByLabel('Target symbol')).toHaveValue('8');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 1.00×' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(await page.locator('.attention-scroll').evaluate(element => element.scrollWidth > element.clientWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('trained-mobile.png'), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.getByLabel('Prepared experiment').selectOption('effect');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('trained-desktop.png'), fullPage: true });
  await page.getByLabel('Model', { exact: true }).selectOption('verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await expect(page.getByText('Random weights, real computations.', { exact: false })).toBeVisible();
  await expect(page.getByLabel('Target symbol')).toHaveValue('2');
  await page.getByLabel('Model', { exact: true }).selectOption('retrieval');
  await expect(page.getByTestId('recipe-note')).toContainText('Ablation changes the answer');
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(errors).toEqual([]);
});
