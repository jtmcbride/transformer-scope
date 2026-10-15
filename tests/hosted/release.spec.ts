import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type { ValidationReport } from '../../src/validation/report';
import textFixture from '../fixtures/text-inference.json' with { type: 'json' };
import continuationFixture from '../fixtures/continuation-reference.json' with { type: 'json' };

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
  for (const name of ['verification', 'retrieval', 'tinystories']) {
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
  await page.goto('?view=details&model=retrieval');
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

test('hosted default flow edits a head and restores the computed prediction', async ({ page }, testInfo) => {
  await page.goto('?model=retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×', { timeout: 60_000 });
  const flow = page.getByRole('region', { name: 'Visual transformer flow', exact: true });
  await flow.getByRole('button', { name: 'Normal selected head', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(page.locator('.visual-answer strong')).toHaveText('3');
  await flow.getByRole('button', { name: 'Mute selected head', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await expect(page.locator('.visual-answer strong')).toHaveText('0');
  await expect(page.getByTestId('target-probability')).toHaveText('99.99% → 0.00%');
  await flow.getByRole('button', { name: 'Restore original prediction', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  await expect(page.getByLabel('Exact head strength')).toHaveValue('1');
  await expect(page.getByTestId('baseline-answer')).toHaveText(await page.getByTestId('modified-answer').innerText());
  await page.getByRole('button', { name: 'Visual flow', exact: true }).click();
  await expect(page.getByTestId('target-probability')).toHaveText('99.99% → 99.99%');
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.locator('.visual-flow').screenshot({ path: testInfo.outputPath('hosted-visual-flow.png') });
});

test('hosted TinyStories default accepts real text and exports verified trained-model predictions', async ({ page }, testInfo) => {
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await expect(page.getByLabel('Model', { exact: true })).toHaveValue('tinystories');
  await expect(page.locator('.visual-answer strong')).toHaveText('she');
  const example = textFixture.cases[1];
  await page.getByLabel('Story prompt').fill(example.text);
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  await page.getByRole('button', { name: 'Mute selected head', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByText('Save, replay & runtime settings', { exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = JSON.parse(await readFile((await (await download).path())!, 'utf8'));
  expect(saved.input).toEqual({ text: example.text, tokenIds: example.tokenIds });
  expect(saved.checkpoint.modelRevision).toBe(textFixture.modelRevision);
  example.variants[0].logits.forEach((value, i) => expect(Math.abs(saved.results.modifiedLogits[i] - value)).toBeLessThanOrEqual(2e-5 + 1e-5 * Math.abs(value)));
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.screenshot({ path: testInfo.outputPath('hosted-tinystories.png'), fullPage: true });
});

test('hosted continuation generates the independent original and edited story token by token', async ({ page }, testInfo) => {
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByLabel('Story prompt').fill(continuationFixture.prompt);
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  await page.getByRole('button', { name: 'Mute selected head', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByLabel('New-token limit').fill('16');
  await page.getByLabel('Compare with current head edit').check();
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 90_000 });
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download continuation', exact: true }).click();
  const record = JSON.parse(await readFile((await (await pending).path())!, 'utf8'));
  expect(record.checkpoint.modelRevision).toBe(continuationFixture.modelRevision);
  expect(record.prompt.tokenIds).toEqual(continuationFixture.promptTokenIds);
  for (const key of ['original', 'edited'] as const) {
    expect(record.frame[key].tokenIds).toEqual(continuationFixture[key].tokenIds);
    record.frame[key].choices.forEach((choice: { probability: number }, i: number) => expect(choice.probability).toBeCloseTo(continuationFixture[key].choices[i].probability, 5));
  }
  await testInfo.attach('hosted-continuation-record', { body: JSON.stringify(record, null, 2), contentType: 'application/json' });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.locator('.story-continuation').screenshot({ path: testInfo.outputPath('hosted-continuation.png') });
  await page.getByRole('button', { name: /^Inspect edited token 7:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  expect(parseFloat(await inspection.getByTestId('selected-token-probability').innerText()) / 100).toBeCloseTo(continuationFixture.edited.choices[6].probability, 5);
  await expect(inspection.getByLabel('Context before selected token')).toHaveText(`${continuationFixture.prompt} play with it. He asked`);
  await expect(inspection.locator('.visual-token-stream > span')).toHaveCount(continuationFixture.promptTokenIds.length + 6);
  await inspection.getByRole('button', { name: 'L1 · Head 1', exact: true }).click();
  await expect(inspection.getByTestId('node-values').locator('dd').nth(1)).toHaveText('0.0000');
  await inspection.screenshot({ path: testInfo.outputPath('hosted-token-inspection.png') });
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
