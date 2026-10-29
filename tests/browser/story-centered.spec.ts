import { expect, test } from '@playwright/test';

test('story playback comes first and a selected token owns the only flow inspector', async ({ page }, info) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  const flow = page.getByRole('region', { name: 'Visual transformer flow', exact: true });
  const story = page.getByRole('article', { name: 'Original story', exact: true });
  expect(await story.evaluate(element => element.getBoundingClientRect().top)).toBeLessThan(700);
  expect(await story.locator('.story-prose').evaluate(element => element.getBoundingClientRect().top)).toBeLessThan(700);
  expect(await page.getByRole('button', { name: 'Play story', exact: true }).evaluate(element => element.getBoundingClientRect().bottom)).toBeLessThan(600);
  await expect(flow).toHaveCount(1);
  expect(await flow.evaluate(element => element.getBoundingClientRect().top)).toBeGreaterThan(await story.evaluate(element => element.getBoundingClientRect().bottom));
  await page.screenshot({ path: info.outputPath('story-first-desktop.png') });
  await page.getByLabel('New-token limit').fill('2');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  await page.getByRole('button', { name: /^Inspect original token 1:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await expect(inspection.getByRole('region', { name: 'Scan heads at selected token', exact: true })).toHaveCount(0);
  await expect(inspection.getByRole('region', { name: 'Scan head pairs at selected token', exact: true })).toHaveCount(0);
  await expect(inspection.getByRole('region', { name: 'Head-pair interaction heatmap', exact: true })).toHaveCount(0);
  await expect(inspection.getByRole('button', { name: 'Compare from this token', exact: true })).toBeEnabled();
  await expect(flow).toHaveCount(1);
  await expect(page.getByRole('complementary', { name: 'Selected node' })).toHaveCount(0);
  await expect(inspection.getByLabel('Active story inspection')).toContainText('Step 1');
  await inspection.getByRole('button', { name: 'L1 · Head 2', exact: true }).click();
  await expect(inspection.getByLabel('Active story inspection')).toContainText('Layer 1 · Head 2');
  await inspection.getByRole('button', { name: 'Next token', exact: false }).click();
  await expect(inspection.getByLabel('Active story inspection')).toContainText('Step 2');
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await expect(flow).toHaveCount(1);
  await inspection.getByRole('button', { name: 'Close token inspection', exact: true }).click();
  await expect(page.locator('.prompt-flow-inspector')).toBeVisible();
  await expect(flow).toHaveCount(1);
  await expect(page.getByRole('button', { name: /^Inspect original token 2:/ })).toBeFocused();
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Computation workspace', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Visual flow', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: /^Inspect original token 1:/ }).click();
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath('story-token-mobile.png') });
});

test('analysis is requested explicitly, closing cancels work and reopening retains partial results', async ({ page }) => {
  test.setTimeout(120_000);
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='COMPARE')setTimeout(()=>dispatch.call(self,e),120);else dispatch.call(self,e);};}` });
  });
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByLabel('New-token limit').fill('2');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  await page.getByRole('button', { name: /^Inspect original token 1:/ }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  const branch = page.getByRole('button', { name: 'Compare from this token', exact: true });
  for (const [kind, scanName, start, listName, status] of [
    ['heads', 'Scan heads at selected token', 'Scan all heads', 'Ranked head changes', 'head-scan-status'],
    ['pairs', 'Scan head pairs at selected token', 'Scan head pairs', 'Ranked head pair changes', 'pair-scan-status'],
  ] as const) {
    const disclosure = page.locator(`.story-analysis-${kind}`), scan = page.getByRole('region', { name: scanName, exact: true });
    await disclosure.locator('summary').click();
    await expect(scan).toBeVisible();
    if (kind === 'pairs') await expect(scan.getByRole('region', { name: 'Head-pair interaction heatmap', exact: true })).toHaveCount(0);
    await scan.getByRole('button', { name: start, exact: true }).click();
    const list = scan.getByRole('list', { name: listName, exact: true });
    await expect(list.locator('li').first()).toBeVisible({ timeout: 60_000 });
    await disclosure.locator('summary').click();
    await expect(branch).toBeEnabled();
    const count = await disclosure.locator('ol > li').count();
    const stoppedStatus = await disclosure.getByTestId(status).textContent();
    await page.waitForTimeout(300);
    expect(await disclosure.locator('ol > li').count()).toBe(count);
    expect(await disclosure.getByTestId(status).textContent()).toBe(stoppedStatus);
    await disclosure.locator('summary').click();
    await expect(scan.getByTestId(status)).toContainText('Results below are partial.');
    await expect(list.locator('li')).toHaveCount(count);
    await disclosure.locator('summary').click();
  }
  await page.getByRole('button', { name: 'Next token →', exact: true }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByLabel('Active story inspection')).toContainText('Step 2');
  await expect(page.getByRole('region', { name: 'Scan heads at selected token', exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Scan head pairs at selected token', exact: true })).toHaveCount(0);
});
