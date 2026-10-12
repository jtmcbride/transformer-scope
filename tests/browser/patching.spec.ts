import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

const fixture = JSON.parse(readFileSync('tests/fixtures/residual-reference.json', 'utf8')).patching as {
  sourceText: string; sweep: { layer: number; stage: string; position: number; logits: number[] }[] };
const softmax = (logits: number[]) => { const m = Math.max(...logits), e = logits.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); };
async function modifiedPercent(page: Page, symbol: string) {
  const row = page.locator('.probability-row').filter({ has: page.locator('.symbol', { hasText: new RegExp(`^${symbol}$`, 'u') }) });
  return Number((await row.locator('.numbers strong').innerText()).replace('%', ''));
}

test('residual patching from another input matches the independent PyTorch sweep', async ({ page }) => {
  await page.goto('');
  const status = page.locator('.status');
  await expect(status).toHaveText('Showing 0.00×', { timeout: 30_000 });
  await page.getByLabel('Intervention type').selectOption('patch');
  // The suggested source asks for another key in the same bindings; it differs only at position 14.
  await expect(page.getByLabel('Source input (same number of symbols)')).toHaveValue(fixture.sourceText);
  await expect(status).toHaveText('Showing patched residual');
  await expect(page.getByLabel('Patch position')).toHaveValue('14');
  await expect(page.getByLabel('Patch position').locator('option:checked')).toHaveText('Position 14: B ← A');
  await expect(page.getByTestId('execution-strategy')).toHaveText('Full forward pass');
  await expect(page.locator('.patched-row .vector-list > div')).toHaveCount(64);
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeDisabled();
  for (const [hook, position] of [['0:block.input', 14], ['0:attention.residual', 13], ['1:block.input', 14], ['1:block.input', 15], ['1:attention.residual', 14], ['1:attention.residual', 15]] as const) {
    await page.getByLabel('Patch point').selectOption(hook);
    await page.getByLabel('Patch position').selectOption(String(position));
    await expect(page.locator('.trace strong')).toContainText(`position ${position}`);
    await expect(status).toHaveText('Showing patched residual');
    const [layer, stage] = hook.split(':');
    const expected = softmax(fixture.sweep.find(entry => entry.layer === Number(layer) && entry.stage === stage && entry.position === position)!.logits);
    for (const [symbol, token] of [['3', 8], ['0', 5]] as const)
      expect(Math.abs(await modifiedPercent(page, symbol) - expected[token] * 100), `${hook}@${position} ${symbol}`).toBeLessThanOrEqual(.0051);
    // Final-layer replacements at an earlier position have no path to the prediction.
    await expect(page.getByText('This setting has no active computational path', { exact: false })).toHaveCount(hook === '1:attention.residual' && position === 14 ? 1 : 0);
  }
  // A CPU worker restart re-creates the source run and keeps the patch settings.
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.getByLabel('Backend')).toHaveValue('cpu');
  await expect(status).toHaveText('Showing patched residual');
  await expect(page.getByLabel('Patch point')).toHaveValue('1:attention.residual');
  await expect(page.getByLabel('Patch position')).toHaveValue('15');
  expect(await modifiedPercent(page, '0')).toBeGreaterThan(99);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByLabel('Source input (same number of symbols)').fill('A = 0 ; ? A =');
  await page.getByRole('button', { name: 'Use source input' }).click();
  await expect(page.getByRole('alert')).toContainText('same number of symbols as the input (15)');
  await expect(status).toHaveText('Showing patched residual');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(status).toHaveText('Showing 1.00×');
  await expect(page.getByLabel('Intervention type')).toHaveValue('head');
  await expect(page.locator('#strength')).toBeVisible();
});
