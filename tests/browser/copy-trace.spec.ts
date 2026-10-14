import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import fixture from '../fixtures/residual-trace-reference.json' with { type: 'json' };

async function recordCommands(page: Page) {
  await page.addInitScript(() => {
    const Original = window.Worker, commands: string[] = [];
    Object.assign(window, { __copyCommands: commands });
    window.Worker = class extends Original {
      postMessage(message: unknown, options: Transferable[] | StructuredSerializeOptions = []) {
        commands.push((message as { command: string }).command);
        if (Array.isArray(options)) super.postMessage(message, options); else super.postMessage(message, options);
      }
    };
  });
}
const calls = (page: Page, command: string) => page.evaluate(cmd => (window as unknown as { __copyCommands: string[] }).__copyCommands.filter(item => item === cmd).length, command);
const ready = (page: Page) => expect(page.getByTestId('trace-step')).toContainText('Step 1 of');
async function openCopy(page: Page, point = '1:block.input', position = 14) {
  await page.goto('?backend=cpu&view=details&model=retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByRole('button', { name: 'Inspect Layer 2 Layer input', exact: true }).click();
  await page.getByRole('button', { name: 'Choose source and position', exact: true }).click();
  await page.getByLabel('Patch point', { exact: true }).selectOption(point);
  await page.getByRole('button', { name: `Copy vector at position ${position}:`, exact: false }).click();
}
async function copy(page: Page) {
  await page.getByRole('button', { name: 'Copy from source', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing patched residual');
}

test('copy choices stay drafts; an input copy enters same-layer attention and reuses captured vectors during navigation', async ({ page }, testInfo) => {
  await recordCommands(page);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await openCopy(page);
  const compares = await calls(page, 'COMPARE'), baselines = await calls(page, 'BASELINE');
  await page.getByRole('button', { name: 'Use current input as source', exact: true }).click();
  await expect(page.locator('.copy-token-pair.different-symbol')).toHaveCount(0);
  await page.getByRole('button', { name: 'Use alternate query', exact: true }).click();
  await expect(page.locator('.copy-token-pair.different-symbol')).toHaveCount(1);
  await page.getByRole('button', { name: 'Inspect Layer 1 Residual after attention', exact: true }).click();
  await expect(page.getByTestId('copy-draft')).toContainText('Layer 1 after attention');
  await page.getByRole('button', { name: 'Inspect Layer 2 Layer input', exact: true }).click();
  await expect(page.getByTestId('copy-draft')).toContainText('Layer 2 input · position 14');
  expect(await calls(page, 'COMPARE')).toBe(compares);
  expect(await calls(page, 'BASELINE')).toBe(baselines);
  expect(await calls(page, 'TRACE')).toBe(0);
  await expect(page.getByTestId('modified-answer')).toHaveText('0'); // The completed head ablation is still displayed.
  await copy(page);
  await expect(page.getByTestId('completed-copy')).toContainText('source A → current B');
  await expect(page.getByTestId('baseline-answer')).toHaveText('3');
  await expect(page.getByTestId('modified-answer')).toHaveText('0');
  await expect(page.getByRole('button', { name: 'Predict after position 15:', exact: false })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click(); await ready(page);
  await expect(page.getByTestId('trace-step')).toContainText('Layer 2 · Layer input · position 14');
  await expect(page.getByTestId('graph-copy-connection')).toContainText('Source residual at position 14');
  await expect(page.getByTestId('trace-handoff')).toContainText('Position 14 → 15');
  await expect(page.getByTestId('trace-path')).toContainText('through Layer 2 attention');
  await expect(page.getByTestId('copied-residual')).toContainText('Source input: A → current input: B');
  const row = fixture.cases.find(item => item.name === 'copy-input-effect')!.residualRow;
  const expectedNorm = (values: number[]) => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  const numbers = await page.getByTestId('node-values').locator('dd').allInnerTexts();
  expect(Math.abs(Number(numbers[0]) - expectedNorm(row.baselineValues))).toBeLessThan(.0006);
  expect(Math.abs(Number(numbers[1]) - expectedNorm(row.copiedValues))).toBeLessThan(.0006);
  const inspections = await calls(page, 'INSPECT'), comparisons = await calls(page, 'COMPARE');
  await page.getByRole('button', { name: 'Inspect full vector', exact: true }).click();
  const cells = await page.getByRole('table', { name: 'Selected node vector dimensions' }).locator('tbody tr td:nth-child(3)').allInnerTexts();
  expect(cells).toHaveLength(64);
  cells.forEach((value, i) => expect(Math.abs(Number(value) - row.copiedValues[i])).toBeLessThan(2e-5 + 1e-5 * Math.abs(row.copiedValues[i])));
  await page.getByRole('button', { name: 'Hide full vector', exact: true }).click();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByTestId('trace-step')).toContainText('Normalize for attention · position 14');
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByTestId('trace-step')).toContainText('Head 1 output · position 15');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('node-readout')).toContainText('Model prediction');
  expect(await calls(page, 'TRACE')).toBe(1);
  expect(await calls(page, 'COMPARE')).toBe(comparisons);
  expect(await calls(page, 'INSPECT')).toBe(inspections);
  await page.getByRole('button', { name: 'Inspect Layer 2 Layer input', exact: true }).click();
  await page.getByLabel('Trace inspection position').selectOption('14');
  await page.getByRole('button', { name: 'Inspect Layer 2 Layer input', exact: true }).evaluate(element => element.scrollIntoView({ block: 'center' }));
  await page.screenshot({ path: testInfo.outputPath('copy-desktop.png') });
});

test('after-attention copies stay local until a later attention layer and cannot travel back to an earlier prediction', async ({ page }) => {
  await recordCommands(page);
  await openCopy(page, '1:attention.residual'); await copy(page);
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click(); await ready(page);
  await expect(page.getByTestId('trace-step')).toContainText('Step 1 of 4');
  await expect(page.getByTestId('trace-path')).toContainText('no computational path');
  await expect(page.getByTestId('trace-handoff')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Residual after attention', exact: true })).toHaveAttribute('data-trace-route', 'local');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 1', exact: true })).toHaveAttribute('data-trace-change', 'exact');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Join heads + project', exact: true })).toHaveAttribute('data-trace-change', 'exact');
  await expect(page.getByTestId('copied-residual')).toContainText('Attention in this layer was computed before the copy');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('readout-delta')).toHaveText('Δ log p: 0 nats');
  const compares = await calls(page, 'COMPARE'), baselines = await calls(page, 'BASELINE');
  await page.getByLabel('Patch point', { exact: true }).selectOption('0:attention.residual');
  expect(await calls(page, 'COMPARE')).toBe(compares);
  await copy(page); await ready(page);
  expect(await calls(page, 'BASELINE')).toBe(baselines); // The same source run supplies another junction.
  await expect(page.getByTestId('trace-handoff')).toContainText('Position 14 → 15');
  await expect(page.getByTestId('trace-path')).toContainText('through Layer 2 attention');
  await expect(page.getByRole('button', { name: 'Inspect Layer 1 Head 1', exact: true })).toHaveAttribute('data-trace-change', 'exact');
  await page.getByRole('button', { name: 'Predict after position 2:', exact: false }).click(); await ready(page);
  await expect(page.locator('.trace-scope')).toContainText('Prediction position 2');
  await expect(page.getByTestId('trace-path')).toContainText('no computational path');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('readout-delta')).toHaveText('Δ log p: 0 nats');
  await page.getByRole('button', { name: 'Reset to baseline' }).click(); await ready(page);
  await expect(page.getByTestId('graph-copy-connection')).toHaveCount(0);
  await expect(page.locator('.flow-node[data-trace-change=changed], .flow-node[data-trace-change=small]')).toHaveCount(0);
});

test('identical source copies stay exact; export, replay and restart preserve the completed source instead of drafts', async ({ page }) => {
  await recordCommands(page);
  await openCopy(page, '1:attention.residual', 15); await copy(page);
  await expect(page.getByTestId('modified-answer')).toHaveText('0');
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click(); await ready(page);
  const compares = await calls(page, 'COMPARE');
  await page.getByRole('button', { name: 'Use current input as source', exact: true }).click();
  expect(await calls(page, 'COMPARE')).toBe(compares);
  await expect(page.getByTestId('modified-answer')).toHaveText('0');
  await copy(page); await ready(page);
  await expect(page.getByTestId('trace-copy-match')).toContainText('exactly matches');
  await expect(page.getByTestId('modified-answer')).toHaveText('3');
  await expect(page.locator('.flow-node[data-trace-change=changed], .flow-node[data-trace-change=small]')).toHaveCount(0);
  await page.getByLabel('Source input (same number of symbols)').fill('A = 0 ; B = 3 ; C = 0 ; ? C =');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = await readFile((await (await download).path())!, 'utf8'), document = JSON.parse(saved);
  expect(document.observation.source.tokenIds).toEqual(document.input.tokenIds);
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.locator('.status')).toHaveText('Showing patched residual'); await ready(page);
  await expect(page.getByLabel('Source input (same number of symbols)')).toHaveValue(document.observation.source.text);
  await expect(page.getByTestId('trace-copy-match')).toContainText('exactly matches');
  await page.getByRole('button', { name: 'Reset to baseline' }).click(); await ready(page);
  await page.getByLabel('Import experiment').setInputFiles({ name: 'copy.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results'); await ready(page);
  await expect(page.getByTestId('trace-copy-match')).toContainText('exactly matches');
  await page.getByLabel('Binding 1 value').selectOption('2');
  await page.getByRole('button', { name: 'Run input' }).click(); await ready(page);
  await expect(page.getByTestId('graph-copy-connection')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Copy a residual vector', exact: true })).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('late source preparation and trace replies cannot overwrite a new checkpoint or worker', async ({ page }) => {
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `{
      const send = self.postMessage.bind(self), delayed = new Set(); let baselines = 0;
      self.addEventListener('message', event => {
        if (event.data?.command === 'LOAD') baselines = 0;
        if (event.data?.command === 'TRACE' || event.data?.command === 'BASELINE' && ++baselines > 1) delayed.add(event.data.id);
      });
      self.postMessage = (message, ...args) => delayed.has(message?.id)
        ? setTimeout(() => { delayed.delete(message.id); send(message, ...args); }, 500) : send(message, ...args);
    }\n${await response.text()}` });
  });
  await openCopy(page);
  await page.getByRole('button', { name: 'Copy from source', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Preparing source input…');
  await page.getByLabel('Model', { exact: true }).selectOption('verification');
  await expect(page.locator('.status')).toHaveText('Showing 0.50×');
  await expect(page.getByRole('region', { name: 'Copy a residual vector', exact: true })).toHaveCount(0);
  await page.getByLabel('Intervention type').selectOption('patch'); await copy(page);
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click(); await ready(page);
  await expect(page.getByTestId('trace-copy-match')).toContainText('exactly matches');
  await page.getByLabel('Source input (same number of symbols)').fill('A E I C H');
  await page.getByRole('button', { name: 'Copy from source', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Preparing source input…');
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.locator('.status')).toHaveText('Showing patched residual'); await ready(page);
  await expect(page.getByLabel('Source input (same number of symbols)')).toHaveValue('A E I C G');
  await expect(page.getByTestId('trace-copy-match')).toContainText('exactly matches');
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('aligned copy positions and guided steps work by keyboard on narrow screens', async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const width of [360, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await openCopy(page);
    const position = page.getByRole('button', { name: 'Copy vector at position 14:', exact: false });
    await position.focus(); await page.keyboard.press('Enter');
    await expect(position).toHaveAttribute('aria-pressed', 'true');
    expect(await page.locator('.copy-alignment').evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    if (width === 390) {
      await page.locator('.copy-alignment').evaluate(element => element.scrollIntoView({ block: 'center' }));
      await page.screenshot({ path: testInfo.outputPath('copy-inputs-mobile.png') });
    }
    await page.getByRole('button', { name: 'Copy from source', exact: true }).focus(); await page.keyboard.press('Enter');
    await expect(page.locator('.status')).toHaveText('Showing patched residual');
    await page.getByRole('button', { name: 'Follow the change', exact: true }).focus(); await page.keyboard.press('Enter'); await ready(page);
    await page.getByRole('button', { name: 'Next', exact: true }).focus(); await page.keyboard.press('Enter');
    await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
    for (const selector of ['.trace-navigation', '.prediction-summary']) expect(await page.locator(selector).evaluate(element => {
      const box = element.getBoundingClientRect(); return box.left >= -1 && box.right <= innerWidth + 1 && box.top >= 0 && box.bottom <= innerHeight;
    })).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    if (width === 390) await page.screenshot({ path: testInfo.outputPath('copy-trace-mobile.png') });
  }
});
