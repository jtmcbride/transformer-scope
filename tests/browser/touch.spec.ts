import { expect, test } from '@playwright/test';

// Touch/viewport emulation: layout and hit-area evidence only, not physical-device evidence.
test.use({ viewport: { width: 360, height: 740 }, hasTouch: true });

test('emulated touch layout keeps controls finger-sized and the page width contained', async ({ page }, testInfo) => {
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×', { timeout: 30_000 });
  await page.getByRole('button', { name: 'Inspect projection, residual, and MLP' }).click();
  await expect(page.getByRole('table', { name: 'Baseline and modified computation vectors' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360);
  const coarse = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
  if (!coarse) { testInfo.annotations.push({ type: 'pointer-not-coarse', description: 'This engine does not report a coarse pointer under touch emulation; hit-area sizes were not asserted.' }); return; }
  // WCAG 2.2 target size (minimum) is 24 CSS px; inline text links are exempt.
  const small = await page.evaluate(() => [...document.querySelectorAll('button, select, input, textarea, summary, [tabindex]')]
    .map(element => { const box = element.getBoundingClientRect(); return { label: (element.getAttribute('aria-label') ?? element.textContent ?? element.id).trim().slice(0, 40), width: box.width, height: box.height }; })
    .filter(box => box.width > 0 && (box.width < 24 || box.height < 24)));
  expect(small).toEqual([]);
  await page.locator('#strength').scrollIntoViewIfNeeded();
  const box = (await page.locator('#strength').boundingBox())!;
  expect(box.height).toBeGreaterThanOrEqual(32);
  await page.touchscreen.tap(box.x + box.width * .75, box.y + box.height / 2);
  await expect(page.locator('.status')).toHaveText(/^Showing 1\.[4-6]\d×$/u);
});
