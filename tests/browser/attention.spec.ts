import { readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';

const routing = JSON.parse(readFileSync('tests/fixtures/attention-reference.json', 'utf8')).routing as {
  sweep: { layer: number; head: number; position: number; logits: number[] }[] };
const softmax = (logits: number[]) => { const m = Math.max(...logits), e = logits.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); };
async function modifiedPercent(page: Page, symbol: string) {
  const row = page.locator('.probability-row').filter({ has: page.locator('.symbol', { hasText: new RegExp(`^${symbol}$`, 'u') }) });
  return Number((await row.locator('.numbers strong').innerText()).replace('%', ''));
}

async function roundTrip(page: Page) {
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = await readFile((await (await download).path())!, 'utf8');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await page.getByLabel('Import experiment').setInputFiles({ name: 'saved.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results', { timeout: 30_000 });
  return JSON.parse(saved);
}

test('redirecting a head\'s attention matches the independent PyTorch routing sweep', async ({ page }) => {
  await page.goto('');
  const status = page.locator('.status');
  await expect(status).toHaveText('Showing 0.00×', { timeout: 30_000 });
  await page.getByLabel('Intervention type').selectOption('attention');
  await expect(status).toHaveText('Showing redirected attention');
  await expect(page.getByLabel('Query position')).toHaveValue('15');
  await expect(page.getByLabel('Attend to')).toHaveValue('uniform');
  await page.getByLabel('Layer', { exact: true }).selectOption('1');
  await page.getByLabel('Head', { exact: true }).selectOption('1');
  for (const position of [3, 5, 14, 0]) {
    await page.getByLabel('Attend to').selectOption(String(position));
    await expect(page.locator('.trace strong')).toHaveText(`Layer 2, head 2 · query 15 · all on position ${position}`);
    await expect(status).toHaveText('Showing redirected attention');
    const expected = softmax(routing.sweep.find(entry => entry.layer === 1 && entry.head === 1 && entry.position === position)!.logits);
    for (const [symbol, token] of [['3', 8], ['0', 5]] as const)
      expect(Math.abs(await modifiedPercent(page, symbol) - expected[token] * 100), `→${position} ${symbol}`).toBeLessThanOrEqual(.0051);
    // The attention matrix shows the replaced row exactly.
    const row = await page.locator('.attention-row').nth(15).locator('.cell').allInnerTexts();
    expect(row).toEqual(Array.from({ length: 16 }, (_, i) => i === position ? '1.00' : '0.00'));
  }
  await expect(page.getByTestId('execution-strategy')).toHaveText('Full forward pass');
  await expect(page.locator('.vector-list > div')).toHaveCount(16);
  // A saved redirect restores its mode and controls, and replays to the same logits.
  await page.getByLabel('Attend to').selectOption('3');
  await expect(page.locator('.trace strong')).toHaveText('Layer 2, head 2 · query 15 · all on position 3');
  const saved = await roundTrip(page);
  expect(saved.schemaVersion).toBe(2);
  expect(saved.observation.interventions[0]).toMatchObject({ kind: 'attention.override', layer: 1, head: 1, query: 15 });
  await expect(status).toHaveText('Showing redirected attention');
  await expect(page.getByLabel('Intervention type')).toHaveValue('attention');
  await expect(page.getByLabel('Attend to')).toHaveValue('3');
  await expect(page.locator('.trace strong')).toHaveText('Layer 2, head 2 · query 15 · all on position 3');
  // Moving the query before the chosen source clamps the focus; a final-layer earlier query cannot reach the prediction.
  await page.getByLabel('Query position').selectOption('2');
  await expect(page.getByLabel('Attend to')).toHaveValue('2');
  await page.getByLabel('Attend to').selectOption('uniform');
  await expect(page.getByLabel('Attend to').locator('option')).toHaveCount(4);
  await expect(status).toHaveText('Showing redirected attention');
  await expect(page.getByText('This setting has no active computational path', { exact: false })).toBeVisible();
  expect(await page.locator('.attention-row').nth(2).locator('.cell').allInnerTexts()).toEqual(['0.33', '0.33', '0.33', ...Array(13).fill('·')]);
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(status).toHaveText('Showing 1.00×');
  await expect(page.getByLabel('Intervention type')).toHaveValue('head');
});
