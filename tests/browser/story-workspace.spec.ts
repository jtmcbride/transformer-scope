import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
async function download(page: Page) {
  const wait = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download workspace', exact: true }).click();
  return JSON.parse(await readFile((await (await wait).path())!, 'utf8'));
}
test('a partially generated saved story resumes its exact sampling sequence after reload', async ({ page }) => {
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByLabel('Continuation sampling').selectOption('sample');
  await page.getByLabel('New-token limit').fill('4');
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.reload();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  const resumed = await download(page);
  await page.getByRole('button', { name: 'Reset story', exact: true }).click();
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  expect((await download(page)).root.frame).toEqual(resumed.root.frame);
});
test('storage failure is visible and the complete workspace remains downloadable', async ({ page }) => {
  await page.addInitScript(() => {
    const set = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, value) { if (key.startsWith('transformer-story-workspace:')) throw new DOMException('Quota exceeded', 'QuotaExceededError'); return set.call(this, key, value); };
  });
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await expect(page.getByRole('alert')).toContainText('Automatic saving is unavailable');
  expect((await download(page)).root.frame.step).toBe(1);
});
