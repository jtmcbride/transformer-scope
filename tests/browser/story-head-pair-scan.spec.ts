import { openProbabilityDetails } from '../support/probability-details';
import { openStoryAnalysis } from '../support/story-analysis';
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
  await openStoryAnalysis(page);
}
const pairName = (edits: { layer: number; head: number; factor: number }[]) => `Use pair layer ${edits[0].layer+1} head ${edits[0].head+1} strength ${edits[0].factor} and layer ${edits[1].layer+1} head ${edits[1].head+1} strength ${edits[1].factor}`;
const pairKey = (edit: { layer: number; head: number; factor: number }) => `${edit.layer}:${edit.head}:${edit.factor}`;
test('full pair scan maps and ranks interactions, then branches from a heatmap cell and saves the verified story', async ({ page }, info) => {
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
  const expected = reference.cases[0], sorted = [...expected.rows].sort((a,b) => Math.abs(b.interaction)-Math.abs(a.interaction));
  await expect(scan.getByLabel('Sort pair scan results')).toHaveValue('interaction-absolute');
  const map = scan.getByRole('region', { name: 'Head-pair interaction heatmap', exact: true });
  await expect(map.locator('.pair-map-cell')).toHaveCount(240);
  await expect(map.locator('.pair-map-pending')).toHaveCount(0); await expect(map.locator('.pair-map-diagonal')).toHaveCount(16);
  const cells = await map.locator('.pair-map-cell').evaluateAll(nodes => nodes.map(node => ({ r:Number(node.getAttribute('data-map-row')), c:Number(node.getAttribute('data-map-column')), interaction:Number(node.getAttribute('data-interaction')), key:node.getAttribute('data-pair-key') })));
  for (const cell of cells) {
    const row = expected.rows.find(row => row.edits.map(pairKey).join('|') === cell.key)!;
    expect(row).toBeTruthy();
    expect(Math.abs(cell.interaction-row.interaction)).toBeLessThanOrEqual(4*5e-6);
    expect(Math.min(cell.r,cell.c)).toBe(row.edits[0].layer*4+row.edits[0].head);
    expect(Math.max(cell.r,cell.c)).toBe(row.edits[1].layer*4+row.edits[1].head);
  }
  expect(await map.locator('.pair-map-cell[tabindex="0"]').count()).toBe(1);
  const firstCell = map.locator('button[data-map-row="0"][data-map-column="1"]'); await firstCell.focus();
  await page.keyboard.press('ArrowDown'); await expect(map.locator('button[data-map-row="2"][data-map-column="1"]')).toBeFocused();
  await page.keyboard.press('Home'); await expect(map.locator('button[data-map-row="2"][data-map-column="0"]')).toBeFocused();
  await page.keyboard.press('Control+Home'); await expect(firstCell).toBeFocused();
  await page.keyboard.press('End'); await expect(map.locator('button[data-map-row="0"][data-map-column="15"]')).toBeFocused();
  const list = scan.getByRole('list', { name: 'Ranked head pair changes' });
  for (let i=0; i<8; i++) { await expect(list.locator('li').nth(i)).toHaveAttribute('data-pair-first', pairKey(sorted[i].edits[0])); await expect(list.locator('li').nth(i)).toHaveAttribute('data-pair-second', pairKey(sorted[i].edits[1])); }
  await page.setViewportSize({ width:390, height:844 });
  expect(await scan.evaluate(el => el.scrollWidth <= el.clientWidth+1 && el.getBoundingClientRect().right <= innerWidth)).toBe(true);
  expect(await map.locator('.pair-map-scroll').evaluate(el => el.scrollWidth > el.clientWidth && el.getBoundingClientRect().right <= innerWidth)).toBe(true);
  await map.screenshot({ path: info.outputPath('mobile-head-pair-map.png') });
  if (info.project.use.hasTouch) {
    expect(await page.evaluate(() => matchMedia('(pointer: coarse)').matches)).toBe(true);
    const size = await firstCell.boundingBox(); expect(size!.width).toBeGreaterThanOrEqual(44); expect(size!.height).toBeGreaterThanOrEqual(44);
  }
  await page.setViewportSize({ width:1440, height:1100 });
  await scan.screenshot({ path: info.outputPath('ranked-head-pairs.png') });
  await scan.getByRole('button', { name: 'Show all 120 pairs', exact: true }).click();
  for (const row of expected.rows) {
    const actual = list.locator(`li[data-pair-first="${pairKey(row.edits[0])}"][data-pair-second="${pairKey(row.edits[1])}"]`);
    expect(parseFloat(await actual.getByTestId('pair-scan-after').innerText())/100).toBeCloseTo(row.after, 5);
    expect(parseFloat(await actual.getByTestId('pair-scan-delta').innerText())/100).toBeCloseTo(row.after-expected.source.choice.probability, 5);
    expect(parseFloat(await actual.getByTestId('pair-scan-sum').innerText())/100).toBeCloseTo(row.sumDelta,5);
    expect(parseFloat(await actual.getByTestId('pair-scan-first-delta').innerText())/100).toBeCloseTo(row.individualProbabilities[0]-expected.source.choice.probability,5);
    expect(parseFloat(await actual.getByTestId('pair-scan-second-delta').innerText())/100).toBeCloseTo(row.individualProbabilities[1]-expected.source.choice.probability,5);
    // Four measured probabilities contribute to I; each keeps its five-decimal parity check.
    expect(Math.abs(parseFloat(await actual.getByTestId('pair-scan-interaction').innerText())/100-row.interaction)).toBeLessThanOrEqual(4*5e-6+0.5e-6);
  }
  for (const mode of ['absolute','increase','decrease','interaction-absolute','interaction-positive','interaction-negative']) {
    await scan.getByLabel('Sort pair scan results').selectOption(mode);
    const interaction = mode.startsWith('interaction-');
    const values = (await list.getByTestId(interaction ? 'pair-scan-interaction' : 'pair-scan-delta').allTextContents()).map(parseFloat);
    const scores = values.map(n => mode.endsWith('absolute') ? Math.abs(n) : mode === 'decrease' || mode === 'interaction-negative' ? -n : n);
    expect(scores.every((n,i) => !i || n <= scores[i-1]+1e-10)).toBe(true);
  }
  const a = sorted[0].edits[0].layer*4+sorted[0].edits[0].head, b = sorted[0].edits[1].layer*4+sorted[0].edits[1].head;
  await map.locator(`button[data-map-row="${a}"][data-map-column="${b}"]`).focus(); await page.keyboard.press('Enter');
  await expect(map.getByRole('button', { pressed: true })).toHaveCount(2);
  await expect(page.getByRole('region', { name: 'Scanned head probability preview', exact: true })).toHaveCount(0);
  expect(Math.abs(parseFloat(await map.getByTestId('heatmap-selected-interaction').innerText())/100-sorted[0].interaction)).toBeLessThanOrEqual(4*5e-6+0.5e-6);
  await map.screenshot({ path: info.outputPath('selected-head-pair-map.png') });
  await map.getByRole('button', { name: 'Preview and use this pair', exact: true }).focus(); await page.keyboard.press('Enter');
  const preview = page.getByRole('region', { name: 'Scanned head probability preview', exact: true });
  await openProbabilityDetails(preview);
  await expect(preview.getByTestId('combined-probability')).toBeVisible({ timeout: 60000 });
  expect(parseFloat(await preview.getByTestId('combined-probability').innerText())/100).toBeCloseTo(sorted[0].after, 5);
  expect(Math.abs(parseFloat(await preview.getByTestId('head-interaction').locator('strong').innerText())/100-sorted[0].interaction)).toBeLessThanOrEqual(4*5e-6+0.5e-6);
  await page.getByLabel('Branch new-token limit').fill('6');
  await page.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60000 });
  const wait = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download branch comparison', exact: true }).click();
  const saved = JSON.parse(await readFile((await (await wait).path())!, 'utf8'));
  expect(saved.schemaVersion).toBe(3); expect(saved.plan.edit).toEqual(expected.bestInteraction.plan.edit); expect(saved.plan.additionalEdits).toEqual(expected.bestInteraction.plan.additionalEdits);
  for (const branch of ['original','edited'] as const) { expect(saved.frame[branch].tokenIds).toEqual(expected.bestInteraction[branch].tokenIds); saved.frame[branch].choices.forEach((choice: { probability: number }, i: number) => expect(choice.probability).toBeCloseTo(expected.bestInteraction[branch].choices[i].probability,5)); }
  await page.reload(); await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
  const report = page.getByRole('region', { name: 'Head change probabilities', exact: true });
  await openProbabilityDetails(report);
  await expect(report.getByTestId('combined-probability')).toBeVisible({ timeout: 60000 });
  expect(parseFloat(await report.getByTestId('combined-probability').innerText())/100).toBeCloseTo(sorted[0].after,5);
  await expect(page.getByRole('navigation', { name: 'Story branch map' }).getByRole('button', { name: /^(Source story$|Comparison \d+:)/ })).toHaveCount(2);
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
  const map = scan.getByRole('region', { name: 'Head-pair interaction heatmap', exact: true });
  await expect(map).toHaveCount(0);
  const first = page.waitForEvent('console', message => message.text() === 'pair-compare');
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click(); await first;
  await expect(map.getByLabel('Heatmap strength assignment')).toHaveValue('0:2');
  await scan.getByRole('button', { name: 'Stop pair scan', exact: true }).click();
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Pair scan stopped. Results below are partial. 0 / 240 combinations measured.');
  await expect(map.locator('.pair-map-cell')).toHaveCount(0);
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click();
  const list = scan.getByRole('list', { name: 'Ranked head pair changes' });
  await expect(list.locator('li').nth(1)).toBeVisible({ timeout: 60000 });
  await scan.getByRole('button', { name: 'Stop pair scan', exact: true }).click();
  await expect(scan.getByTestId('pair-scan-status')).toContainText('Results below are partial.');
  const rows = reference.cases[3].rows, firstRow = list.locator('li').first();
  const a = await firstRow.getAttribute('data-pair-first'), b = await firstRow.getAttribute('data-pair-second');
  const expected = rows.find(row => pairKey(row.edits[0]) === a && pairKey(row.edits[1]) === b)!;
  expect(parseFloat(await firstRow.getByTestId('pair-scan-after').innerText())/100).toBeCloseTo(expected.after,5);
  const scannedInteraction = parseFloat(await firstRow.getByTestId('pair-scan-interaction').innerText());
  expect(Math.abs(scannedInteraction/100-expected.interaction)).toBeLessThanOrEqual(4*5e-6+0.5e-6);
  await firstRow.getByRole('button', { name: pairName(expected.edits), exact: true }).click();
  const chosenKey = expected.edits.map(pairKey).join('|');
  await expect(map.locator(`button[data-pair-key="${chosenKey}"]`)).toHaveAttribute('aria-pressed','true');
  const oldRow = await map.locator(`button[data-pair-key="${chosenKey}"]`).getAttribute('data-map-row');
  const oldColumn = await map.locator(`button[data-pair-key="${chosenKey}"]`).getAttribute('data-map-column');
  await map.getByLabel('Heatmap strength assignment').selectOption('2:0');
  const transposed = map.locator(`button[data-pair-key="${chosenKey}"]`);
  await expect(transposed).toHaveAttribute('data-map-row',oldColumn!); await expect(transposed).toHaveAttribute('data-map-column',oldRow!);
  await expect(transposed).toHaveAttribute('aria-pressed','true');
  expect(Number(await transposed.getAttribute('data-interaction'))).toBeCloseTo(scannedInteraction/100,6);
  const preview = page.getByRole('region', { name: 'Scanned head probability preview', exact: true });
  await openProbabilityDetails(preview);
  await expect(preview.getByTestId('combined-probability')).toBeVisible({ timeout: 60000 });
  expect(parseFloat(await preview.getByTestId('combined-probability').innerText())/100).toBeCloseTo(expected.after,5);
  expect(parseFloat(await preview.getByTestId('head-interaction').locator('strong').innerText())).toBeCloseTo(scannedInteraction,4);
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
  await expect(map).toHaveCount(0);
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click();
  await expect(list.locator('li').nth(1)).toBeVisible({ timeout: 60000 });
  await scan.getByRole('button', { name: 'Stop pair scan', exact: true }).click();
  await expect(map.getByLabel('Heatmap strength assignment').locator('option')).toHaveCount(9);
  await map.getByLabel('Heatmap strength assignment').selectOption('2:1');
  await map.getByLabel('Heatmap strength assignment').selectOption('0:0');
  await map.locator('button[data-map-row="0"][data-map-column="1"]').click();
  await map.getByLabel('Heatmap strength assignment').selectOption('2:2');
  await expect(map.getByTestId('heatmap-outside-view')).toBeVisible();
  await expect(map.locator('.pair-map-cell')).toHaveCount(0);
  await expect(map.locator('.pair-map-pending')).toHaveCount(240);
  pending = page.waitForEvent('console', message => message.text() === 'pair-compare');
  await scan.getByRole('button', { name: 'Scan head pairs', exact: true }).click(); await pending;
  await page.getByRole('button', { name: 'Next token →', exact: true }).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({ timeout: 60000 });
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Ready to scan. 0 / 120 combinations measured.');
  await expect(page.getByRole('button', { name: 'Scan all heads', exact: true })).toBeEnabled();
  await expect(map).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});
