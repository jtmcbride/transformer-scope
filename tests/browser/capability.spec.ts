import { expect, test } from '@playwright/test';

test('an unavailable worker WebGL context produces a clear error and CPU recovery', async ({ page }) => {
  // Emulate the actual missing worker capability; CPU inference still uses the real production bundle.
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    const script = await response.text();
    const capability = `if (typeof OffscreenCanvas !== 'undefined') {
      const originalContext = OffscreenCanvas.prototype.getContext;
      OffscreenCanvas.prototype.getContext = function(kind, ...args) {
        if (kind.includes('webgl')) return null;
        return originalContext.call(this, kind, ...args);
      };
    }\n`;
    await route.fulfill({ response, body: capability + script });
  });
  await page.goto('?model=verification');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Backend').selectOption('webgl');
  await expect(page.getByRole('alert')).toContainText('WebGL worker unavailable:');
  await page.getByRole('button', { name: 'Load with CPU' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await expect(page.getByLabel('Backend')).toHaveValue('cpu');
  await expect(page.getByRole('alert')).toHaveCount(0);
});
