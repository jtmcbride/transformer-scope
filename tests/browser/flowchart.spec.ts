import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import ablations from '../fixtures/ablation-reference.json' with { type: 'json' };
import inspection from '../fixtures/inspection.json' with { type: 'json' };
import type { ExperimentDocument } from '../../src/engine/experiment';

async function exported(page: Page): Promise<ExperimentDocument> {
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  return JSON.parse(await readFile((await (await download).path())!, 'utf8'));
}

test('graph navigation preserves the intervention; a head preset applies to the selected head and matches PyTorch', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('?view=details');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const before = await exported(page);
  await page.getByRole('button', { name: 'Inspect Layer 2 Head 2', exact: true }).click();
  await expect(page.getByRole('complementary', { name: 'Selected node' })).toContainText('Layer 2 · Head 2');
  await expect(page.getByLabel('Exact head strength')).toHaveValue('1');
  await expect(page.getByTestId('node-values')).toBeVisible();
  const inspected = await exported(page);
  expect(inspected.observation).toEqual(before.observation);
  expect(inspected.results).toEqual(before.results);
  // Keyboard activation uses the same contextual controls as a pointer.
  await page.getByRole('group', { name: 'Head strength presets' }).getByRole('button', { name: 'Mute' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('Exact head strength')).toHaveValue('0');
  const applied = await exported(page);
  expect(applied.observation.interventions[0]).toMatchObject({ kind: 'head.scale', layer: 1, head: 1, factor: 0, tokens: { kind: 'all' } });
  const reference = ablations.inputs.find(input => input.text === applied.input.text)!.single.find(row => row.layer === 1 && row.head === 1)!;
  applied.results.modifiedLogits.forEach((value, token) => expect(Math.abs(value - reference.logits[token])).toBeLessThanOrEqual(2e-5 + 1e-5 * Math.abs(reference.logits[token])));
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(page.getByTestId('baseline-answer')).toHaveText(await page.getByTestId('modified-answer').innerText());
  await page.screenshot({ path: testInfo.outputPath('flow-desktop.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('the structured input stays a draft until run, and the exported input reflects the executed bindings', async ({ page }) => {
  await page.goto('?view=details');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByRole('group', { name: 'Head strength presets' }).getByRole('button', { name: 'Normal' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  const before = await exported(page);
  await page.getByLabel('Binding 1 value').selectOption('2');
  await page.getByLabel('Ask for', { exact: true }).selectOption('A');
  await expect(page.getByText('Draft · run to update the input', { exact: true })).toBeVisible();
  expect((await exported(page)).input).toEqual(before.input);
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.getByTestId('baseline-answer')).toHaveText('2');
  await expect(page.getByTestId('modified-answer')).toHaveText('2');
  expect((await exported(page)).input.text).toBe('A = 2 ; B = 3 ; C = 0 ; ? A =');
  await page.getByRole('button', { name: 'Edit tokens', exact: true }).click();
  await page.getByLabel('Space-separated symbols').fill('unknown');
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.getByRole('alert')).toContainText('Unknown symbol');
  await expect(page.getByTestId('baseline-answer')).toHaveText('2');
});

test('stage nodes inspect actual completed vectors without applying an intervention', async ({ page }) => {
  await page.goto('?model=verification&view=details');
  await expect(page.locator('.status')).toHaveText('Showing 0.50×');
  await page.getByLabel('Exact head strength').fill('0.37');
  await page.getByLabel('Exact head strength').press('Enter');
  await expect(page.locator('.status')).toHaveText('Showing 0.37×');
  await page.getByRole('button', { name: 'Inspect Layer 1 Residual after attention', exact: true }).click();
  const reference = inspection.vectors['0']['attention.residual'];
  const norm = (values: number[]) => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  const values = page.getByTestId('node-values').locator('dd');
  await expect(values).toHaveCount(3);
  expect(Math.abs(Number(await values.nth(0).innerText()) - norm(reference.baseline))).toBeLessThan(.0001);
  expect(Math.abs(Number(await values.nth(1).innerText()) - norm(reference.modified))).toBeLessThan(.0001);
  await expect(page.locator('.status')).toHaveText('Showing 0.37×');
  await page.getByRole('button', { name: 'Inspect Input embeddings', exact: true }).click();
  await expect(values).toHaveCount(3);
  await expect(values.nth(2)).toHaveText('0.00e+0');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByRole('complementary', { name: 'Selected node' })).toContainText('Softmax converts');
});

test('the result remains in the mobile viewport while exploring and resetting', async ({ page }, testInfo) => {
  for (const width of [360, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('?view=details');
    await expect(page.locator('.status')).toHaveText('Showing 0.00×');
    const summary = page.getByLabel('Prediction summary');
    const contained = async () => summary.evaluate(element => {
      const box = element.getBoundingClientRect();
      return box.left >= 0 && box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight;
    });
    expect(await contained()).toBe(true);
    await page.getByRole('button', { name: 'Inspect Layer 2 Head 2', exact: true }).click();
    await page.getByRole('group', { name: 'Head strength presets' }).getByRole('button', { name: 'Mute' }).click();
    await expect(page.getByTestId('baseline-answer')).toBeVisible();
    expect(await contained()).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await page.getByRole('button', { name: 'Reset to baseline' }).click();
    await expect(page.locator('.status')).toHaveText('Showing 1.00×');
    expect(await contained()).toBe(true);
    if (width === 390) await page.screenshot({ path: testInfo.outputPath('flow-mobile.png') });
  }
});

test('pending edits keep the completed prediction and intervention visibly identified', async ({ page }) => {
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `${await response.text()}\n{
      const dispatch = self.onmessage;
      self.onmessage = event => event.data?.command === 'COMPARE'
        ? setTimeout(() => dispatch.call(self, event), 500)
        : dispatch.call(self, event);
    }` });
  });
  await page.goto('?view=details');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const summary = page.getByLabel('Prediction summary');
  const previous = await page.getByTestId('target-probability').innerText();
  await page.getByRole('group', { name: 'Head strength presets' }).getByRole('button', { name: 'Normal' }).click();
  await expect(summary).toHaveAttribute('aria-busy', 'true');
  await expect(summary).toContainText('Last completed result · updating');
  await expect(page.getByTestId('target-probability')).toHaveText(previous);
  await expect(page.getByRole('button', { name: 'Inspect Layer 1 Head 4', exact: true })).toContainText('Edited here');
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeDisabled();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(summary).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByTestId('target-probability')).toHaveText('99.99% → 99.99%');
  await expect(page.getByRole('button', { name: 'Inspect Layer 1 Head 4', exact: true })).not.toContainText('Edited here');
});
