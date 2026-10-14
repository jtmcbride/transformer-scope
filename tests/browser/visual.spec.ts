import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';

test('visual flow is the default and view switching preserves completed experiments', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('?model=retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const flow = page.getByRole('region', { name: 'Visual transformer flow', exact: true });
  await expect(flow).toBeVisible();
  await expect(page.getByRole('button', { name: 'Visual flow', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('region', { name: 'Computation workspace', exact: true })).toBeHidden();
  const initial = await page.getByTestId('target-probability').innerText();
  await flow.getByRole('button', { name: 'L2 · Head 2', exact: true }).click();
  await expect(page.getByTestId('target-probability')).toHaveText(initial);
  await expect(page.getByRole('complementary', { name: 'Selected node' })).toContainText('Layer 2 · Head 2');
  await flow.getByRole('button', { name: 'Mute selected head', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await expect(flow.getByRole('button', { name: 'L2 · Head 2', exact: true })).toContainText('Your change');
  const modified = await page.getByTestId('target-probability').innerText();
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  await expect(page.getByTestId('target-probability')).toHaveText(modified);
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 2', exact: true })).toHaveAttribute('aria-pressed', 'true');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = JSON.parse(await readFile((await (await download).path())!, 'utf8'));
  expect(saved.observation.interventions[0]).toMatchObject({ kind: 'head.scale', layer: 1, head: 1, factor: 0 });
  await page.getByRole('button', { name: 'Visual flow', exact: true }).click();
  await expect(page.getByTestId('target-probability')).toHaveText(modified);
  await flow.getByRole('button', { name: 'Restore original prediction', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(page.getByTestId('baseline-answer')).toHaveText(await page.getByTestId('modified-answer').innerText());
  await page.setViewportSize({ width: 1440, height: 1050 });
  await page.screenshot({ path: testInfo.outputPath('visual-desktop.png'), fullPage: true });
  await page.locator('.visual-flow').screenshot({ path: testInfo.outputPath('visual-flow.png') });
  expect(errors).toEqual([]);
});

test('visual flow fits mobile, supports keyboard selection, and keeps draft input separate', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('?model=retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const region = page.getByRole('region', { name: 'Scrollable model flow' });
  expect(await region.evaluate(element => element.scrollWidth > element.clientWidth)).toBe(true);
  const head = page.getByRole('button', { name: 'L1 · Head 1', exact: true });
  await head.focus(); await page.keyboard.press('Enter');
  await expect(head).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('group', { name: 'Change the selected head' })).toBeVisible();
  expect(await page.getByRole('group', { name: 'Change the selected head' }).evaluate(element => {
    const box = element.getBoundingClientRect(); return box.bottom <= document.querySelector('.visual-flow')!.getBoundingClientRect().bottom;
  })).toBe(true);
  const tokens = (await page.locator('.visual-token-stream').textContent())!;
  await page.getByLabel('Binding 1 value').selectOption('2');
  await expect(page.locator('.visual-token-stream')).toHaveText(tokens);
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.locator('.visual-token-stream')).not.toHaveText(tokens);
  await page.getByText('Where did the edited head look?', { exact: false }).click();
  await expect(page.getByRole('img', { name: /Largest attention weights/ })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('visual-mobile.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('inline controls retain the completed result during rapid edits and preserve a selected position', async ({ page }) => {
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `${await response.text()}\n{
      const dispatch = self.onmessage;
      self.onmessage = event => event.data?.command === 'COMPARE'
        ? setTimeout(() => dispatch.call(self, event), 500)
        : dispatch.call(self, event);
    }` });
  });
  await page.goto('?model=retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const flow = page.getByRole('region', { name: 'Visual transformer flow', exact: true });
  const previous = await page.locator('.visual-answer strong').innerText();
  await flow.getByRole('button', { name: 'Normal selected head' }).click();
  await expect(flow).toHaveAttribute('aria-busy', 'true');
  await expect(page.locator('.visual-answer')).toContainText('Last completed answer');
  await expect(page.locator('.visual-answer strong')).toHaveText(previous);
  await expect(page.locator('.visual-control-note')).toContainText('Updating');
  await flow.getByRole('button', { name: 'Amplify selected head' }).click();
  await flow.getByRole('button', { name: 'Mute selected head' }).click();
  await expect(flow).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByLabel('Selected head output strength')).toHaveValue('0');
  await expect(page.getByLabel('Exact head strength')).toHaveValue('0');
  await page.getByLabel('Apply at', { exact: true }).selectOption('0');
  await expect(flow).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('group', { name: 'Change the selected head' })).toContainText('Applies at position 0');
  await page.getByLabel('Selected head output strength').fill('0.37');
  await expect(page.locator('.status')).toHaveText('Showing 0.37×');
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  await expect(page.getByLabel('Apply at', { exact: true })).toHaveValue('0');
  await expect(page.getByLabel('Exact head strength')).toHaveValue('0.37');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = JSON.parse(await readFile((await (await download).path())!, 'utf8'));
  expect(saved.observation.interventions[0]).toMatchObject({ kind: 'head.scale', factor: 0.37, tokens: { kind: 'indices', indices: [0] } });
});
