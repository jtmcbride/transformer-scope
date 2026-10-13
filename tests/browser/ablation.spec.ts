import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

const fixture = JSON.parse(readFileSync('tests/fixtures/ablation-reference.json', 'utf8')) as { inputs: {
  text: string; targetTokenId: number; baselineLogits: number[]; single: { layer: number; head: number; logits: number[] }[]; pairs: { heads: number[][]; logits: number[] }[] }[] };
const logSoftmax = (logits: number[]) => { const m = Math.max(...logits), z = Math.log(logits.reduce((a, v) => a + Math.exp(v - m), 0)); return logits.map(v => v - m - z); };
// Four displayed decimals: half a unit of rounding plus twice the 2e-5 logit bound on a log-probability difference.
const bound = 5e-5 + 4e-5 + 1e-6;

test('the ablation explorer ranks independent PyTorch head effects and exposes a non-additive pair', async ({ page }) => {
  const input = fixture.inputs[1];
  const target = input.targetTokenId, base = logSoftmax(input.baselineLogits)[target];
  const expected = input.single.map(entry => ({ name: `Layer ${entry.layer + 1}, Head ${entry.head + 1}`, delta: logSoftmax(entry.logits)[target] - base,
    probability: Math.exp(logSoftmax(entry.logits)[target]) })).sort((a, b) => a.delta - b.delta);
  await page.goto('');
  const status = page.locator('.status');
  await expect(status).toHaveText('Showing 0.00×', { timeout: 30_000 });
  await page.getByLabel('Prepared experiment').selectOption('small');
  await expect(page.locator('#sequence')).toHaveValue(input.text);
  await expect(status).toHaveText('Showing 0.00×');
  await expect(page.getByRole('table', { name: 'Single-head ablations' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Ablate each head for this prediction' }).click();
  const rows = page.getByRole('table', { name: 'Single-head ablations' }).locator('tbody tr');
  await expect(rows).toHaveCount(8);
  const shown = await rows.evaluateAll(elements => elements.map(row => [row.querySelector('th')!.textContent!, ...Array.from(row.querySelectorAll('td'), cell => cell.textContent!.trim())]));
  // Rows are ordered by measured effect; near-ties may order differently, so compare each head's own values.
  expect(shown[0][0]).toBe(expected[0].name);
  for (const row of shown) {
    const reference = expected.find(item => item.name === row[0])!;
    expect(Math.abs(Number(row[1]) - reference.delta), row[0]).toBeLessThanOrEqual(bound);
    expect(Math.abs(Number(row[2].replace('%', '')) - reference.probability * 100), row[0]).toBeLessThanOrEqual(.0051);
  }
  // Two first-layer heads with negligible single effects change the prediction substantially together.
  await page.getByLabel('Select Layer 1, Head 2 for a pair').check();
  await page.getByLabel('Select Layer 1, Head 4 for a pair').check();
  await expect(page.getByLabel('Select Layer 2, Head 2 for a pair')).toBeDisabled();
  await page.getByRole('button', { name: 'Ablate the selected pair together' }).click();
  const pair = input.pairs.find(item => JSON.stringify(item.heads) === JSON.stringify([[0, 1], [0, 3]]))!;
  const joint = logSoftmax(pair.logits)[target] - base;
  const result = page.getByTestId('pair-result');
  await expect(result).toContainText('Layer 1, Head 2 + Layer 1, Head 4: together');
  const numbers = (await result.locator('strong').allInnerTexts()).map(Number);
  expect(Math.abs(numbers[0] - joint)).toBeLessThanOrEqual(bound);
  expect(numbers[0]).toBeLessThan(-5);
  expect(numbers[1]).toBeGreaterThan(-.01);
  // A row can be shown as the main experiment.
  await page.getByRole('button', { name: 'Show Layer 2, Head 2 ablated' }).click();
  await expect(status).toHaveText('Showing 0.00×');
  await expect(page.getByLabel('Layer', { exact: true })).toHaveValue('1');
  await expect(page.getByLabel('Head', { exact: true })).toHaveValue('1');
  await expect(page.getByLabel('Apply at')).toHaveValue('all');
  await expect(rows).toHaveCount(8);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
