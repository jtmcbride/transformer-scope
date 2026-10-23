import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import reference from '../fixtures/story-head-pair-scan-reference.json' with { type: 'json' };
async function ready(page: Page, edited = false) {
  await page.goto('?backend=cpu'); await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60000 });
  await page.getByLabel('Story prompt').fill('Tom found a red ball. He wanted to');
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  if (edited) { await page.getByRole('button', { name: 'Mute selected head', exact: true }).click(); await expect(page.locator('.status')).toHaveText('Showing 0.00×'); await page.getByLabel('Compare with current head edit').check(); }
  await page.getByLabel('New-token limit').fill('8'); await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60000 });
  await page.getByRole('button', { name: edited ? /^Inspect edited token 7:/ : /^Inspect original token 6:/ }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
}
const pairName = (edits: { layer: number; head: number; factor: number }[]) => `Use pair layer ${edits[0].layer+1} head ${edits[0].head+1} strength ${edits[0].factor} and layer ${edits[1].layer+1} head ${edits[1].head+1} strength ${edits[1].factor}`;
const pairKey = (edit: { layer: number; head: number; factor: number }) => `${edit.layer}:${edit.head}:${edit.factor}`;
test('full pair scan ranks real joint effects, selects the strongest pair, and saves its verified story branch', async ({ page }, info) => {
  test.setTimeout(240000); await ready(page);
  const scan = page.getByRole('region', { name: 'Scan head pairs at selected token', exact: true });
  for (const [mode, total] of [['all',1080],['opposite',240],['normal',120],['amplify',120],['mute-normal',240],['normal-amplify',240],['mute',120]] as const) {
    await scan.getByLabel('Pair scan strengths').selectOption(mode);
    await expect(scan.getByTestId('pair-scan-status')).toHaveText(`Ready to scan. 0 / ${total} combinations measured.`);
  }
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Scan all heads', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Compare from this token', exact: true })).toBeDisabled();
  await expect(scan.getByLabel('Pair scan strengths')).toBeDisabled();
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Pair scan complete. 120 / 120 combinations measured.', { timeout: 150000 });
  const expected = reference.cases[0], sorted = [...expected.rows].sort((a,b) => Math.abs(b.after-expected.source.choice.probability)-Math.abs(a.after-expected.source.choice.probability));
  const list = scan.getByRole('list', { name: 'Ranked head pair changes' });
  for (let i=0; i<8; i++) { await expect(list.locator('li').nth(i)).toHaveAttribute('data-pair-first', pairKey(sorted[i].edits[0])); await expect(list.locator('li').nth(i)).toHaveAttribute('data-pair-second', pairKey(sorted[i].edits[1])); }
  await page.setViewportSize({ width:390, height:844 });
  expect(await scan.evaluate(el => el.scrollWidth <= el.clientWidth+1 && el.getBoundingClientRect().right <= innerWidth)).toBe(true);
  await page.setViewportSize({ width:1440, height:1100 });
  await scan.screenshot({ path: info.outputPath('ranked-head-pairs.png') });
  await scan.getByRole('button', { name: 'Show all 120 pairs', exact: true }).click();
  for (const row of expected.rows) {
    const actual = list.locator(`li[data-pair-first="${pairKey(row.edits[0])}"][data-pair-second="${pairKey(row.edits[1])}"]`);
    expect(parseFloat(await actual.getByTestId('pair-scan-after').innerText())/100).toBeCloseTo(row.after, 5);
    expect(parseFloat(await actual.getByTestId('pair-scan-delta').innerText())/100).toBeCloseTo(row.after-expected.source.choice.probability, 5);
  }
  for (const mode of ['increase','decrease']) {
    await scan.getByLabel('Sort pair scan results').selectOption(mode);
    const values = (await list.getByTestId('pair-scan-delta').allTextContents()).map(parseFloat);
    expect(values.every((n,i) => !i || (mode === 'increase' ? n <= values[i-1] : n >= values[i-1]))).toBe(true);
  }
  await scan.getByRole('button', { name: pairName(sorted[0].edits), exact: true }).focus(); await page.keyboard.press('Enter');
  const preview = page.getByRole('region', { name: 'Scanned head probability preview', exact: true });
  await expect(preview.getByTestId('combined-probability')).toBeVisible({ timeout: 60000 });
  expect(parseFloat(await preview.getByTestId('combined-probability').innerText())/100).toBeCloseTo(sorted[0].after, 5);
  await page.getByLabel('Branch new-token limit').fill('6');
  await page.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60000 });
  const wait = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download branch comparison', exact: true }).click();
  const saved = JSON.parse(await readFile((await (await wait).path())!, 'utf8'));
  expect(saved.schemaVersion).toBe(3); expect(saved.plan.edit).toEqual(expected.best.plan.edit); expect(saved.plan.additionalEdits).toEqual(expected.best.plan.additionalEdits);
  for (const branch of ['original','edited'] as const) { expect(saved.frame[branch].tokenIds).toEqual(expected.best[branch].tokenIds); saved.frame[branch].choices.forEach((choice: { probability: number }, i: number) => expect(choice.probability).toBeCloseTo(expected.best[branch].choices[i].probability,5)); }
  await page.reload(); await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
  const report = page.getByRole('region', { name: 'Head change probabilities', exact: true });
  await expect(report.getByTestId('combined-probability')).toBeVisible({ timeout: 60000 });
  expect(parseFloat(await report.getByTestId('combined-probability').innerText())/100).toBeCloseTo(sorted[0].after,5);
  await expect(page.getByRole('navigation', { name: 'Story branch map' }).getByRole('button')).toHaveCount(2);
  await page.getByRole('button', { name: /^Inspect with head change token 6:/ }).click();
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60000 });
  await expect(page.getByRole('alert')).toHaveCount(0);
});
test('partial mixed pair results retain inherited edits, exclude competing scans, and discard work on navigation', async ({ page }) => {
  test.setTimeout(180000);
  await page.route('**/model.worker-*.js', async route => { const response = await route.fetch(); await route.fulfill({ response, body: `${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='COMPARE'){console.info('pair-compare');setTimeout(()=>dispatch.call(self,e),150);}else dispatch.call(self,e);};}` }); });
  await ready(page, true);
  const scan = page.getByRole('region', { name: 'Scan head pairs at selected token', exact: true });
  await scan.getByLabel('Pair scan strengths').selectOption('opposite');
  const first = page.waitForEvent('console', message => message.text() === 'pair-compare');
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click(); await first;
  await scan.getByRole('button', { name: 'Stop pair scan', exact: true }).click();
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Pair scan stopped. Results below are partial. 0 / 240 combinations measured.');
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click();
  const list = scan.getByRole('list', { name: 'Ranked head pair changes' });
  await expect(list.locator('li').nth(1)).toBeVisible({ timeout: 60000 });
  await scan.getByRole('button', { name: 'Stop pair scan', exact: true }).click();
  await expect(scan.getByTestId('pair-scan-status')).toContainText('Results below are partial.');
  const rows = reference.cases[3].rows, firstRow = list.locator('li').first();
  const a = await firstRow.getAttribute('data-pair-first'), b = await firstRow.getAttribute('data-pair-second');
  const expected = rows.find(row => pairKey(row.edits[0]) === a && pairKey(row.edits[1]) === b)!;
  expect(parseFloat(await firstRow.getByTestId('pair-scan-after').innerText())/100).toBeCloseTo(expected.after,5);
  await firstRow.getByRole('button', { name: pairName(expected.edits), exact: true }).click();
  const preview = page.getByRole('region', { name: 'Scanned head probability preview', exact: true });
  await expect(preview.getByTestId('combined-probability')).toBeVisible({ timeout: 60000 });
  expect(parseFloat(await preview.getByTestId('combined-probability').innerText())/100).toBeCloseTo(expected.after,5);
  await page.getByLabel('Branch new-token limit').fill('1'); await page.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60000 });
  let pending = page.waitForEvent('console', message => message.text() === 'pair-compare');
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click(); await pending;
  await page.getByRole('button', { name: /^Inspect with head change token 7:/ }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
  await page.getByRole('button', { name: 'Back to previous comparison', exact: true }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
  await expect(page.getByRole('button', { name: 'Start new branch comparison', exact: true })).toBeEnabled();
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Ready to scan. 0 / 120 combinations measured.');
  await scan.getByLabel('Pair scan strengths').selectOption('all');
  pending = page.waitForEvent('console', message => message.text() === 'pair-compare');
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click(); await pending;
  await page.getByRole('button', { name: 'Next token →', exact: true }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Ready to scan. 0 / 120 combinations measured.');
  await expect(page.getByRole('button', { name: 'Scan all heads', exact: true })).toBeEnabled();
  await expect(page.getByRole('alert')).toHaveCount(0);
});
