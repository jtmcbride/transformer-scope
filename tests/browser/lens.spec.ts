import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

const fixture = JSON.parse(readFileSync('tests/fixtures/lens.json', 'utf8')) as {
  baseline: { logits: number[][]; norms: number[] }; modified: { logits: number[][]; norms: number[] } };
const symbols = (JSON.parse(readFileSync('public/models/retrieval/tokenizer.json', 'utf8')) as { tokens: string[] }).tokens;
function expected(logits: number[]) {
  const maximum = Math.max(...logits), exps = logits.map(value => Math.exp(value - maximum)), sum = exps.reduce((a, b) => a + b, 0);
  const probabilities = exps.map(value => value / sum);
  const top = probabilities.reduce((best, value, token) => value > probabilities[best] ? token : best, 0);
  return { top: symbols[top], topPercent: probabilities[top] * 100, targetPercent: probabilities[symbols.indexOf('3')] * 100 };
}
const percent = (text: string) => Number(text.match(/(-?\d+\.\d+)%/u)![1]);

test('the residual-stream read-out shows independent PyTorch lens values for baseline and ablation', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×', { timeout: 30_000 });
  await expect(page.getByRole('table', { name: 'Residual stream read-out' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Read out the residual stream' }).click();
  const rows = page.getByRole('table', { name: 'Residual stream read-out' }).locator('tbody tr');
  await expect(rows).toHaveCount(5);
  await expect(rows.last().locator('th')).toHaveText('Layer 2 block output · prediction');
  for (let index = 0; index < 5; index++) {
    const cells = await rows.nth(index).locator('td').allInnerTexts();
    const base = expected(fixture.baseline.logits[index]), modified = expected(fixture.modified.logits[index]);
    expect(cells[0].split(' ')[0], `row ${index} baseline top`).toBe(base.top);
    expect(cells[1].split(' ')[0], `row ${index} modified top`).toBe(modified.top);
    // Displayed with two decimals: half a unit of rounding plus a small allowance for float32 differences.
    for (const [text, value] of [[cells[0], base.topPercent], [cells[1], modified.topPercent], [cells[2], base.targetPercent], [cells[3], modified.targetPercent]] as const)
      expect(Math.abs(percent(text) - value), `row ${index}: ${text}`).toBeLessThanOrEqual(.0051);
    expect(Math.abs(Number(cells[4]) - fixture.baseline.norms[index])).toBeLessThanOrEqual(.00051);
    expect(Math.abs(Number(cells[5]) - fixture.modified.norms[index])).toBeLessThanOrEqual(.00051);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  // The wide table scrolls inside its panel; the panel itself stays within the viewport (not clipped by the workspace).
  const scroll = await page.locator('.residual-lens .vector-table-scroll').evaluate(element => ({ right: element.getBoundingClientRect().right, client: element.clientWidth, content: element.scrollWidth }));
  expect(scroll.right).toBeLessThanOrEqual(390);
  expect(scroll.content).toBeGreaterThan(scroll.client);
  // At the baseline strength, the modified read-out equals the baseline read-out.
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(rows.last().locator('td').nth(1)).toHaveText(/^3 /u);
  for (let index = 0; index < 5; index++) {
    const cells = await rows.nth(index).locator('td').allInnerTexts();
    expect([cells[1], cells[3], cells[5]]).toEqual([cells[0], cells[2], cells[4]]);
  }
});
