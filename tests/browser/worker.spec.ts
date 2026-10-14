import { test, expect } from '@playwright/test';

test('production worker matches PyTorch on CPU and WebGL, including gradients and Adam', async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  let workerUrl = '';
  page.on('request', request => { if (/model\.worker-.*\.js/u.test(request.url())) workerUrl = request.url(); });
  await page.goto('?model=verification&view=details');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  expect(workerUrl).not.toBe('');
  const reports = await page.evaluate(async url => {
    const worker = new Worker(url, { type: 'module' });
    let id = 0;
    const pending = new Map<number, { resolve: (result: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = event => {
      const response = event.data;
      const call = pending.get(response.id);
      if (!call) return;
      pending.delete(response.id);
      if (response.ok) call.resolve(response.result);
      else call.reject(new Error(`${response.error.code}: ${response.error.message}`));
    };
    const request = (command: string, payload: unknown) => new Promise<Record<string, number | string>>((resolve, reject) => {
      const current = ++id;
      pending.set(current, { resolve: result => resolve(result as Record<string, number | string>), reject });
      worker.postMessage({ id: current, command, payload });
    });
    try {
      const reports = [];
      for (const backend of ['cpu', 'webgl']) {
        try {
          await request('LOAD', { backend, manifestUrl: new URL('models/verification/manifest.json', window.location.href).href });
          reports.push({ backend, status: 'passed', report: await request('PROBE', {}) });
        } catch (error) {
          if (backend !== 'webgl' || !(error instanceof Error) || !error.message.startsWith('UNSUPPORTED_BACKEND')) throw error;
          reports.push({ backend, status: 'unsupported', reason: error.message });
        }
      }
      return reports;
    } finally { worker.terminate(); }
  }, workerUrl);
  expect(reports[0].status).toBe('passed');
  if (testInfo.project.name === 'chromium') expect(reports[1].status).toBe('passed');
  for (const result of reports) {
    if (result.status !== 'passed') continue;
    const report = result.report!;
    expect(report.retainedTensorDelta).toBe(0);
    expect(report.forwardMaximumError).toBeLessThan(2e-5);
    expect(report.gradientMaximumError).toBeLessThan(3e-5);
    expect(report.optimizerMaximumError).toBeLessThan(3e-5);
  }
  await testInfo.attach('worker-probe-results', { body: JSON.stringify({ userAgent: await page.evaluate(() => navigator.userAgent), reports }, null, 2), contentType: 'application/json' });
});

test('head scaling, rapid edits, reset, and causal controls use computed results', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('?model=verification&view=details');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await expect(page.getByText('Random weights, real computations.', { exact: false })).toBeVisible();
  await expect(page.getByTestId('execution-strategy')).toHaveText('Cached suffix · layer 1');
  await page.screenshot({ path: testInfo.outputPath('desktop.png'), fullPage: true });
  const slider = page.getByRole('slider', { name: 'Head strength', exact: true });
  await slider.fill('0'); await slider.fill('1.5'); await slider.fill('0.25');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.25×' })).toBeVisible();
  const vector = await page.locator('.vector-list > div').evaluateAll(rows => rows.map(row => {
    const values = row.querySelectorAll('code'); return Array.from(values).map(value => Number(value.textContent));
  }));
  vector.forEach(([raw, modified]) => expect(Math.abs(raw * .25 - modified)).toBeLessThan(1e-5));
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 1.00×' })).toBeVisible();
  await expect(page.getByTestId('execution-strategy')).toHaveText('Baseline reused');
  await expect(page.locator('.delta').first()).toHaveText('+0.00 pp');
  await page.getByLabel('Layer', { exact: true }).selectOption('1');
  await page.getByLabel('Apply at').selectOption('0');
  await slider.fill('0');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible();
  await expect(page.getByTestId('execution-strategy')).toHaveText('Cached suffix · layer 2');
  await expect(page.getByText('This setting has no active computational path', { exact: false })).toBeVisible();
  await expect(page.locator('.delta').first()).toHaveText('+0.00 pp');
  await page.getByRole('button', { name: 'Verify against PyTorch' }).click();
  await expect(page.getByText('Reference checks passed')).toBeVisible({ timeout: 20_000 });
  expect(errors).toEqual([]);
});

test('input errors preserve the current experiment and a mobile layout remains usable', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('?model=verification&view=details');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Space-separated symbols').fill('unknown');
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.getByRole('alert')).toContainText('Unknown symbol');
  await page.getByLabel('Space-separated symbols').fill('A B C');
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.getByText('Next token after position', { exact: false })).toContainText('3');
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('mobile.png'), fullPage: true });
});

test('corrupt static weights are rejected before model execution', async ({ page }) => {
  await page.route('**/weights.bin', async route => {
    const response = await route.fetch();
    const data = await response.body(); data[0] ^= 1;
    await route.fulfill({ response, body: data });
  });
  await page.goto('?model=verification&view=details');
  await expect(page.getByRole('alert')).toContainText('checksum mismatch');
  await expect(page.locator('.probability-row')).toHaveCount(0);
});
