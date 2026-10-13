import { expect, test } from '@playwright/test';
import fixture from '../fixtures/inspection.json' with { type: 'json' };

test('exact strength entry and attention-cell arithmetic match independent PyTorch values', async ({ page }) => {
  await page.goto('?model=verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Exact head strength').fill('3');
  await page.getByLabel('Exact head strength').press('Enter');
  await expect(page.getByRole('alert')).toContainText('Use a strength from 0 to 2');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Exact head strength').fill('0.37');
  await page.getByLabel('Exact head strength').press('Enter');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.37×' })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  for (const cell of fixture.cells) {
    await page.getByRole('button', { name: new RegExp(`^Query ${cell.query}, source ${cell.source},`) }).click();
    await expect(page.getByTestId('attention-cell-detail')).toContainText(`Query ${cell.query} → source ${cell.source}`);
    expect(Math.abs(Number(await page.getByTestId('cell-score').textContent()) - cell.score)).toBeLessThan(2e-5);
    expect(Math.abs(Number(await page.getByTestId('cell-weight').textContent()) - cell.weight)).toBeLessThan(2e-5);
    if (cell.source > cell.query) await expect(page.getByTestId('cell-masked-score')).toContainText('-1e9 · future position');
    else expect(Math.abs(Number(await page.getByTestId('cell-masked-score').textContent()) - cell.maskedScore)).toBeLessThan(2e-5);
  }
});

test('projection/residual/MLP selectors show independent completed values, reset, and mobile containment', async ({ page }, testInfo) => {
  await page.goto('?model=verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Exact head strength').fill('0.37'); await page.getByLabel('Exact head strength').press('Enter');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.37×' })).toBeVisible();
  await page.getByRole('button', { name: 'Inspect projection, residual, and MLP' }).click();
  const table = page.getByRole('table', { name: 'Baseline and modified computation vectors' });
  for (const layer of ['0', '1'] as const) {
    await page.getByRole('combobox', { name: 'Inspect layer', exact: true }).selectOption(layer);
    for (const stage of ['attention.projected', 'attention.residual', 'mlp.output', 'block.output'] as const) {
      await page.getByLabel('Computation stage').selectOption(stage);
      await expect(table.locator('tbody tr')).toHaveCount(8);
      // Wait for the selected reference's rows rather than reading a previous slice.
      const reference = fixture.vectors[layer][stage];
      await expect.poll(async () => {
        const rows = await table.locator('tbody tr').evaluateAll(elements => elements.map(row => Array.from(row.querySelectorAll('td')).map(cell => Number(cell.textContent))));
        return Math.max(...rows.flatMap(([baseline, modified], i) => [Math.abs(baseline - reference.baseline[i]), Math.abs(modified - reference.modified[i])]));
      }).toBeLessThan(2e-5);
    }
  }
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 1.00×' })).toBeVisible();
  await expect.poll(async () => table.locator('tbody td:last-child').allTextContents()).toEqual(Array(8).fill('0.000e+0'));
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('inspection-mobile.png'), fullPage: true });
  await page.locator('.computation-inspector').screenshot({ path: testInfo.outputPath('vector-panel-mobile.png') });
});

test('worker restart during edits creates a fresh CPU session and preserves the requested experiment', async ({ page }) => {
  const workers: string[] = [];
  page.on('worker', worker => workers.push(worker.url()));
  await page.goto('?model=verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Layer', { exact: true }).selectOption('1');
  await page.getByLabel('Apply at').selectOption('0');
  await page.getByRole('slider', { name: 'Head strength', exact: true }).fill('0.23');
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.23×' })).toBeVisible();
  await expect(page.getByLabel('Backend')).toHaveValue('cpu');
  await expect(page.getByLabel('Layer', { exact: true })).toHaveValue('1');
  await expect(page.getByLabel('Apply at')).toHaveValue('0');
  await expect(page.getByText('This setting has no active computational path', { exact: false })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(workers).toHaveLength(2);
  await page.getByRole('slider', { name: 'Head strength', exact: true }).fill('1.27');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 1.27×' })).toBeVisible();
});

test('CPU restart can cancel an unfinished model load', async ({ page }) => {
  let first = true;
  let unblock!: () => void;
  const blocked = new Promise<void>(resolve => { unblock = resolve; });
  let loading!: () => void;
  const loadStarted = new Promise<void>(resolve => { loading = resolve; });
  await page.route('**/weights.bin', async route => {
    if (first) { first = false; loading(); await blocked; await route.abort().catch(() => {}); }
    else await route.continue();
  });
  try {
    await page.goto('?model=verification');
    await loadStarted;
    await expect(page.getByRole('status').filter({ hasText: 'Loading model…' })).toBeVisible();
    await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
    await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
    await expect(page.getByRole('alert')).toHaveCount(0);
  } finally { unblock(); }
});
