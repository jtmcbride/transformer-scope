import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';

async function recordCommands(page: Page) {
  await page.addInitScript(() => {
    const Original = window.Worker, commands: string[] = [];
    Object.assign(window, { __redirectCommands: commands });
    window.Worker = class extends Original {
      postMessage(message: unknown, options: Transferable[] | StructuredSerializeOptions = []) {
        commands.push((message as { command: string }).command);
        if (Array.isArray(options)) super.postMessage(message, options); else super.postMessage(message, options);
      }
    };
  });
}
const calls = (page: Page, command: string) => page.evaluate(cmd => (window as unknown as { __redirectCommands: string[] }).__redirectCommands.filter(item => item === cmd).length, command);
const ready = (page: Page) => expect(page.getByTestId('trace-step')).toContainText('Step 1 of');
async function startRedirect(page: Page, source = 5) {
  await page.goto('?backend=cpu&view=details&model=retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByLabel('Intervention type').selectOption('attention');
  await page.getByLabel('Layer', { exact: true }).selectOption('1');
  await page.getByLabel('Head', { exact: true }).selectOption('1');
  // Complete setup before activating a source; pending head changes can move the control panel.
  await expect(page.locator('.completed-redirect > .eyebrow')).toHaveText('Completed redirect · Layer 2 · Head 2');
  await expect(page.locator('.status')).toHaveText('Showing redirected attention');
  const sourceButton = page.getByRole('button', { name: `Attend to position ${source}:`, exact: false });
  await sourceButton.focus(); await page.keyboard.press('Enter');
  await expect(sourceButton).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('redirect-connection')).toContainText(`Source ${source}:`);
  await expect(page.locator('.status')).toHaveText('Showing redirected attention');
}

test('token routing exposes the edited weights and guides the changed output without recomputation', async ({ page }, testInfo) => {
  await recordCommands(page);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await startRedirect(page);
  await expect(page.getByTestId('baseline-answer')).toHaveText('3');
  await expect(page.getByTestId('modified-answer')).toHaveText('0');
  expect(await calls(page, 'TRACE')).toBe(0);
  const compares = await calls(page, 'COMPARE');
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await ready(page);
  await expect(page.getByTestId('trace-step')).toContainText('Head 2 attention weights · position 15');
  await expect(page.getByTestId('graph-redirect-connection')).toContainText('Source 5: B → Query 15: =');
  await expect(page.getByTestId('redirect-scores')).toContainText('Attention scores: Exact match');
  await expect(page.getByTestId('trace-first-flip')).toContainText('Layer 2 after attention');
  const edited = page.getByRole('button', { name: 'Inspect Layer 2 Head 2 attention row', exact: true });
  await expect(edited).toHaveAttribute('data-trace-change', 'changed');
  await expect(edited).toHaveAttribute('data-trace-route', 'prediction');
  await expect(edited).toHaveAttribute('aria-pressed', 'true');
  expect(await edited.evaluate(element => {
    const box = element.getBoundingClientRect();
    return Array.from(element.children).every(child => {
      const bounds = child.getBoundingClientRect(); return bounds.top >= box.top - 1 && bounds.bottom <= box.bottom + 1;
    });
  })).toBe(true);
  const table = page.locator('.node-inspection').getByRole('table', { name: 'Attention weights before and after redirect' });
  await expect(table.locator('tr[data-source-position="5"] td').last()).toHaveText('100.00%');
  await expect(table.locator('tr[data-source-position="14"] td').last()).toHaveText('0.00%');
  const inspections = await calls(page, 'INSPECT');
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
  await expect(page.getByTestId('trace-step')).toContainText('Head 2 output');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 2', exact: true })).toHaveAttribute('data-trace-change', 'changed');
  await page.getByRole('button', { name: 'Inspect Layer 2 Residual after attention', exact: true }).click();
  await expect(page.getByTestId('node-readout')).toContainText('Preferred token: 3 → 0');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('node-readout')).toContainText('Model prediction');
  expect(await calls(page, 'TRACE')).toBe(1);
  expect(await calls(page, 'COMPARE')).toBe(compares);
  expect(await calls(page, 'INSPECT')).toBe(inspections);
  await page.screenshot({ path: testInfo.outputPath('redirect-desktop.png') });
});

test('small effects, query-to-prediction handoffs and local routes stay distinct from the chosen source', async ({ page }) => {
  await startRedirect(page, 14);
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await ready(page);
  await expect(page.getByTestId('modified-answer')).toHaveText('3');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 2 attention row', exact: true })).not.toHaveAttribute('data-trace-change', 'exact');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('readout-delta')).not.toHaveText('Δ log p: 0 nats');
  await page.getByRole('button', { name: 'Inspect Layer 2 Head 2', exact: true }).click();
  await page.getByRole('button', { name: 'Query position 2:', exact: false }).click();
  await page.getByRole('button', { name: 'Attend to position 1:', exact: false }).click();
  await ready(page);
  await expect(page.getByTestId('trace-path')).toContainText('no computational path');
  await expect(page.getByLabel('Trace inspection position')).toHaveValue('2');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 2 attention row', exact: true })).toHaveAttribute('data-trace-route', 'local');
  await page.getByLabel('Trace inspection position').selectOption('1');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 2 attention row', exact: true })).toHaveAttribute('data-trace-change', 'exact');
  await expect(page.getByRole('button', { name: 'Inspect Layer 2 Head 2 attention row', exact: true })).toHaveAttribute('data-trace-route', 'unaffected');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('readout-delta')).toHaveText('Δ log p: 0 nats');
  await page.getByRole('button', { name: 'Inspect Layer 2 Head 2', exact: true }).click();
  await page.getByLabel('Layer', { exact: true }).selectOption('0');
  await page.getByRole('button', { name: 'Query position 3:', exact: false }).click();
  await page.getByRole('button', { name: 'Attend to position 0:', exact: false }).click();
  await ready(page);
  await expect(page.getByTestId('graph-redirect-connection')).toContainText('Source 0:');
  await expect(page.getByTestId('trace-handoff')).toContainText('Position 3 → 15');
  await expect(page.getByTestId('trace-path')).toContainText('through Layer 2 attention');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await ready(page);
  await expect(page.getByTestId('trace-path')).toContainText('no effective edit');
  await expect(page.locator('.flow-node[data-trace-change=changed], .flow-node[data-trace-change=small]')).toHaveCount(0);
  await expect(page.getByTestId('graph-redirect-connection')).toHaveCount(0);
});

test('redirect tokens work by keyboard on narrow screens and future sources cannot be selected', async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const width of [360, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await startRedirect(page);
    await page.getByRole('button', { name: 'Query position 2:', exact: false }).click();
    const sources = page.getByRole('group', { name: 'Attention source', exact: true });
    await expect(sources.getByRole('button', { name: 'Attend to position 5:', exact: false })).toBeDisabled();
    const source = sources.getByRole('button', { name: 'Attend to position 1:', exact: false });
    await source.focus(); await page.keyboard.press('Enter');
    await expect(source).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('redirect-connection')).toContainText('Source 1: A → Query 2: =');
    await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
    await ready(page);
    await page.getByRole('button', { name: 'Next', exact: true }).focus(); await page.keyboard.press('Enter');
    await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
    for (const selector of ['.trace-navigation', '.prediction-summary']) expect(await page.locator(selector).evaluate(element => {
      const box = element.getBoundingClientRect(); return box.left >= -1 && box.right <= innerWidth + 1 && box.top >= 0 && box.bottom <= innerHeight;
    })).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    if (width === 390) await page.screenshot({ path: testInfo.outputPath('redirect-mobile.png') });
  }
});

test('late redirect traces cannot overwrite newer source choices, replay or a CPU restart', async ({ page }) => {
  await recordCommands(page);
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `{
      const send = self.postMessage.bind(self), traces = new Set();
      self.addEventListener('message', event => { if (event.data?.command === 'TRACE') traces.add(event.data.id); });
      self.postMessage = (message, ...args) => traces.has(message?.id)
        ? setTimeout(() => { traces.delete(message.id); send(message, ...args); }, 500)
        : send(message, ...args);
    }\n${await response.text()}` });
  });
  await startRedirect(page);
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await expect.poll(() => calls(page, 'TRACE')).toBe(1);
  await page.getByRole('button', { name: 'Attend to position 3:', exact: false }).click();
  await page.getByRole('button', { name: 'Attend to position 14:', exact: false }).click();
  await ready(page);
  await expect(page.getByTestId('graph-redirect-connection')).toContainText('Source 14: B');
  await expect(page.getByTestId('modified-answer')).toHaveText('3');
  expect(await calls(page, 'TRACE')).toBeLessThanOrEqual(2);
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = await readFile((await (await download).path())!, 'utf8');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await ready(page);
  await page.getByLabel('Import experiment').setInputFiles({ name: 'redirect.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results');
  await ready(page);
  await expect(page.getByTestId('graph-redirect-connection')).toContainText('Source 14: B');
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await ready(page);
  await expect(page.getByTestId('graph-redirect-connection')).toContainText('Source 14: B');
  await expect(page.getByRole('alert')).toHaveCount(0);
});
