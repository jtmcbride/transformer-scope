import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type { ValidationReport } from '../../src/validation/report';

interface Release {
  format: string; schemaVersion: number; commit: string; sourceTreeDirty: boolean; basePath: string;
  models: Record<string, { modelRevision: string }>;
  assets: { path: string; byteLength: number; sha256: string }[];
}

test('hosted release identity, asset bytes, MIME types, and model revisions agree', async ({ request }, testInfo) => {
  // Pages completion can precede CDN visibility; wait for this release, not an older success.
  await expect.poll(async () => {
    const visible = await request.get(`release.json?expected=${process.env.EXPECTED_COMMIT ?? 'current'}`);
    if (visible.status() !== 200) return `HTTP ${visible.status()}`;
    const stamp = await visible.json() as Release;
    return process.env.EXPECTED_COMMIT ? stamp.commit : stamp.format;
  }, { timeout: 60_000, intervals: [1000, 2000, 5000] }).toBe(process.env.EXPECTED_COMMIT ?? 'transformer-microscope-release');
  const response = await request.get(`release.json?expected=${process.env.EXPECTED_COMMIT ?? 'current'}`);
  expect(response.status()).toBe(200);
  expect(response.headers()['content-type']).toContain('application/json');
  const release = await response.json() as Release;
  expect(release.format).toBe('transformer-microscope-release');
  expect(release.schemaVersion).toBe(1);
  expect(release.basePath).toBe('/transformer-scope/');
  expect(release.commit).toMatch(/^[0-9a-f]{40}$/u);
  if (process.env.EXPECTED_COMMIT) expect(release.commit).toBe(process.env.EXPECTED_COMMIT);
  if (new URL(testInfo.project.use.baseURL!).protocol === 'https:') expect(release.sourceTreeDirty).toBe(false);
  expect(release.assets.length).toBeGreaterThanOrEqual(13);
  const paths = new Set<string>();
  const records = [];
  for (const asset of release.assets) {
    expect(asset.path).toMatch(/^[a-zA-Z0-9_/.-]+$/u);
    expect(asset.path.split('/')).not.toContain('..');
    expect(paths.has(asset.path)).toBe(false); paths.add(asset.path);
    expect(asset.sha256).toMatch(/^[a-f0-9]{64}$/u);
    const fetched = await request.get(asset.path);
    expect(fetched.status(), asset.path).toBe(200);
    const bytes = await fetched.body();
    expect(bytes.length, asset.path).toBe(asset.byteLength);
    expect(createHash('sha256').update(bytes).digest('hex'), asset.path).toBe(asset.sha256);
    const type = fetched.headers()['content-type'];
    if (asset.path.endsWith('.js')) expect(type).toMatch(/javascript|ecmascript/u);
    if (asset.path.endsWith('.css')) expect(type).toContain('text/css');
    if (asset.path.endsWith('.json')) expect(type).toContain('application/json');
    if (asset.path.endsWith('.bin')) expect(type).toContain('application/octet-stream');
    if (asset.path === 'index.html') expect(type).toContain('text/html');
    records.push({ path: asset.path, contentType: type, byteLength: bytes.length, sha256: asset.sha256 });
  }
  expect(paths.has('index.html')).toBe(true);
  expect(paths.has('validation.html')).toBe(true);
  expect(paths.has('study.html')).toBe(true);
  expect([...paths].some(path => /assets\/model\.worker-.*\.js/u.test(path))).toBe(true);
  for (const name of ['verification', 'retrieval']) {
    for (const file of ['manifest.json', 'weights.bin', 'tokenizer.json', 'probe.json']) expect(paths.has(`models/${name}/${file}`)).toBe(true);
    const manifest = await (await request.get(`models/${name}/manifest.json`)).json();
    expect(manifest.modelRevision).toBe(release.models[name].modelRevision);
  }
  await testInfo.attach('hosted-asset-audit', { body: JSON.stringify({ url: testInfo.project.use.baseURL, release, records }, null, 2), contentType: 'application/json' });
});

test('hosted worker runs recipes, inspection, reference verification, reset, export/replay, and restart', async ({ page }, testInfo) => {
  const errors: string[] = [], failed: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => failed.push(`${request.url()}: ${request.failure()?.errorText}`));
  await page.goto('');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('recipe-note')).toContainText('Ablation changes the answer');
  await expect(page.getByTestId('execution-strategy')).toHaveText('Cached suffix · layer 1');
  await page.getByRole('button', { name: 'Verify against PyTorch' }).click();
  await expect(page.getByText('Reference checks passed')).toBeVisible({ timeout: 60_000 });
  const reference = await page.locator('.probe-result').innerText();
  await page.getByLabel('Prepared experiment').selectOption('unreachable');
  await expect(page.getByText('This setting has no active computational path', { exact: false })).toBeVisible();
  await page.getByLabel('Prepared experiment').selectOption('small');
  await expect(page.getByTestId('recipe-note')).toContainText('A reachable edit has little effect');
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = await readFile((await (await download).path())!, 'utf8');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 1.00×' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled();
  await page.getByLabel('Import experiment').setInputFiles({ name: 'hosted-experiment.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results');
  await page.getByRole('button', { name: 'Inspect projection, residual, and MLP' }).click();
  await expect(page.getByRole('table', { name: 'Baseline and modified computation vectors' }).locator('tbody tr')).toHaveCount(64);
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.00×' })).toBeVisible();
  await expect(page.getByLabel('Backend')).toHaveValue('cpu');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByLabel('Model', { exact: true }).selectOption('verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  expect(errors).toEqual([]); expect(failed).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('hosted-workspace.png'), fullPage: true });
  await testInfo.attach('hosted-workflow-check', { body: JSON.stringify({ url: page.url(), userAgent: await page.evaluate(() => navigator.userAgent), reference, errors, failed }, null, 2), contentType: 'application/json' });
});

test('hosted opt-in validation harness identifies the release and passes every supported check', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('validation.html');
  await page.getByLabel('Evidence class').selectOption('automation');
  await page.getByLabel('Device description').fill('CI hosted check; not a physical device');
  await page.getByLabel('Verification checkpoint').setChecked(false);
  await page.getByLabel('Warm-up edits').fill('1');
  await page.getByLabel('Measured edits').fill('5');
  await page.getByRole('button', { name: 'Start validation' }).click();
  await expect(page.getByTestId('validation-outcome')).toBeVisible({ timeout: 150_000 });
  const report = JSON.parse(await page.getByTestId('validation-json').textContent() ?? '') as ValidationReport;
  await testInfo.attach('hosted-device-validation', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
  if (process.env.EXPECTED_COMMIT) expect((report.release.data as { commit: string }).commit).toBe(process.env.EXPECTED_COMMIT);
  const failures = report.combinations.flatMap(item => item.checks.filter(check => check.status === 'failed').map(check => `${item.requestedBackend}/${check.name}: ${check.detail}`));
  expect(failures).toEqual([]);
  expect(report.summary.outcome).toBe('passed');
  expect(errors).toEqual([]);
});
