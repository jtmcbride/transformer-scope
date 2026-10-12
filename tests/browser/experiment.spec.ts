import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import { comparePredictions } from '../../src/engine/comparison';
import type { ExperimentDocument } from '../../src/engine/experiment';

test('export and replay restore completed settings, switch checkpoints and compare newly computed predictions', async ({ page }, testInfo) => {
  await page.goto('?model=verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByRole('slider', { name: 'Head strength', exact: true }).fill('0.35');
  await page.getByRole('button', { name: 'Predict after position 2: E' }).click();
  await page.getByLabel('Target symbol').selectOption('14');
  await page.getByLabel('Apply at').selectOption('0');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.35×' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled({ timeout: 30_000 });
  // A textarea draft is not part of the completed experiment.
  await page.getByLabel('Space-separated symbols').fill('unknown');
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const path = await (await downloaded).path();
  const contents = await readFile(path!, 'utf8');
  const document = JSON.parse(contents) as ExperimentDocument;
  expect(document.input.text).toBe('A E I C G');
  expect(document.observation.predictionPosition).toBe(2);
  expect(document.observation.targetTokenId).toBe(14);
  expect(document.observation.interventions[0].tokens).toEqual({ kind: 'indices', indices: [0] });
  expect(document.results.baselineLogits).toHaveLength(16);
  expect(document.results.modifiedLogits).toHaveLength(16);
  expect(document.results.precision.dtype).toBe('float32');
  await page.getByLabel('Model', { exact: true }).selectOption('retrieval');
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled({ timeout: 30_000 });
  await page.getByLabel('Import experiment').setInputFiles({ name: 'saved.json', mimeType: 'application/json', buffer: Buffer.from(contents) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results');
  await expect(page.getByLabel('Model', { exact: true })).toHaveValue('verification');
  await expect(page.getByLabel('Space-separated symbols')).toHaveValue('A E I C G');
  await expect(page.getByRole('slider', { name: 'Head strength', exact: true })).toHaveValue('0.35');
  await expect(page.getByLabel('Target symbol')).toHaveValue('14');
  await expect(page.getByLabel('Apply at')).toHaveValue('0');
  await expect(page.getByText('Next token after position', { exact: false })).toContainText('2');
  await page.getByLabel('Backend').selectOption('webgl');
  const unavailable = page.getByRole('alert').filter({ hasText: 'WebGL worker unavailable:' });
  await expect.poll(async () => await unavailable.isVisible() || await page.getByRole('button', { name: 'Export experiment' }).isEnabled(), { timeout: 30_000 }).toBe(true);
  if (await unavailable.isVisible()) {
    testInfo.annotations.push({ type: 'unsupported-webgl', description: (await unavailable.textContent())! });
    await page.getByRole('button', { name: 'Load with CPU' }).click();
  }
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled({ timeout: 30_000 });
  await page.getByLabel('Import experiment').setInputFiles({ name: 'saved.json', mimeType: 'application/json', buffer: Buffer.from(contents) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.getByTestId('replay-report')).toHaveCount(0);
  await expect(page.getByRole('status').filter({ hasText: 'Showing 1.00×' })).toBeVisible();
});

test('invalid imports preserve the current experiment; self-consistent changed measurements are exposed on replay', async ({ page }) => {
  await page.goto('?model=verification');
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled({ timeout: 30_000 });
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const contents = await readFile((await (await downloaded).path())!, 'utf8');
  const original = JSON.parse(contents) as ExperimentDocument;
  const stale = structuredClone(original); stale.checkpoint.modelRevision = '0'.repeat(64);
  await page.getByLabel('Import experiment').setInputFiles({ name: 'stale.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(stale)) });
  await expect(page.getByRole('alert')).toContainText('saved checkpoint');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  const changed = structuredClone(original);
  changed.results.modifiedLogits = changed.results.modifiedLogits.map(value => value + .1);
  changed.results.metrics = comparePredictions(changed.results.baselineLogits, changed.results.modifiedLogits,
    { schemaVersion: 1, kind: 'symbols', bosTokenId: 0, tokens: ['<BOS>', ...'ABCDEFGHIJKLMNO'] }, 8, changed.observation.targetTokenId).metrics;
  await page.getByLabel('Import experiment').setInputFiles({ name: 'changed.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(changed)) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay differs from saved results');
  await expect(page.getByTestId('replay-report')).toContainText('Showing newly computed predictions');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByLabel('Import experiment').setInputFiles({ name: 'broken.json', mimeType: 'application/json', buffer: Buffer.from('{') });
  await expect(page.getByRole('alert')).toContainText('not valid JSON');
});
