import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import fixture from '../fixtures/trace-reference.json' with { type: 'json' };

const titles = { effect: 'Ablation changes the answer', small: 'A reachable edit has little effect', unreachable: 'An earlier final-layer edit cannot reach the answer' };
async function record(page: Page) {
  await page.addInitScript(() => {
    const Original = window.Worker, commands: string[] = [];
    Object.assign(window, { __exampleCommands: commands });
    window.Worker = class extends Original {
      postMessage(message: unknown, options: Transferable[] | StructuredSerializeOptions = []) {
        commands.push((message as { command: string }).command);
        if (Array.isArray(options)) super.postMessage(message, options); else super.postMessage(message, options);
      }
    };
  });
}
const calls = (page: Page, command: string) => page.evaluate(cmd => (window as unknown as { __exampleCommands: string[] }).__exampleCommands.filter(item => item === cmd).length, command);
const guide = (page: Page) => page.getByTestId('example-guide');

test('each example card recomputes its reference result and connects inspection to one reusable trace', async ({ page }, testInfo) => {
  await record(page);
  await page.setViewportSize({ width: 1440, height: 1100 });
  for (const id of ['effect', 'small', 'unreachable'] as const) {
    await page.goto('?backend=cpu&view=details');
    await expect(page.locator('.status')).toHaveText('Showing 0.00×');
    const reference = fixture.cases.find(item => item.name === id)!;
    const card = page.getByRole('button', { name: `Start example: ${titles[id]}`, exact: true });
    await card.click();
    await expect(guide(page).getByRole('heading')).toHaveText(titles[id]);
    await expect(page.getByRole('button', { name: 'Trace this example', exact: true })).toBeEnabled();
    await expect(card).toHaveAttribute('aria-pressed', 'true');
    const top = (logits: number[]) => logits.reduce((best, value, token) => value > logits[best] ? token : best, 0);
    // The shipped retrieval vocabulary's value symbols occupy IDs 5–8.
    await expect(page.getByTestId('example-result')).toContainText(`${top(reference.baselineLogits) - 5} → ${top(reference.modifiedLogits) - 5}`);
    const probability = (logits: number[], target: number) => {
      const max = Math.max(...logits), exp = logits.map(value => Math.exp(value - max));
      return exp[target] / exp.reduce((sum, value) => sum + value, 0) * 100;
    };
    const target = top(reference.baselineLogits);
    const percentages = [...(await page.getByTestId('example-result').innerText()).matchAll(/([\d.]+)%/gu)].map(match => Number(match[1]));
    expect(Math.abs(percentages[0] - probability(reference.baselineLogits, target))).toBeLessThan(.0051);
    expect(Math.abs(percentages[1] - probability(reference.modifiedLogits, target))).toBeLessThan(.0051);
    if (id === 'small') await expect(page.getByTestId('example-result')).toContainText('e-');
    expect(await calls(page, 'TRACE')).toBe(0);
    const comparisons = await calls(page, 'COMPARE');
    await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect the edited head', exact: true }).click();
    await expect(page.getByRole('button', { name: `Inspect Layer ${reference.intervention.layer + 1} Head ${reference.intervention.head + 1}`, exact: true })).toHaveAttribute('aria-pressed', 'true');
    expect(await calls(page, 'COMPARE')).toBe(comparisons);
    expect(await calls(page, 'TRACE')).toBe(0);
    await page.getByRole('button', { name: 'Trace this example', exact: true }).click();
    await expect(page.getByTestId('trace-step')).toContainText('Step 1 of');
    if (id === 'effect') await expect(page.getByTestId('trace-first-flip')).toContainText('Layer 2 after attention');
    if (id === 'small') await expect(page.getByTestId('trace-handoff')).toContainText('Position 0 → 15');
    if (id === 'unreachable') await expect(page.getByTestId('trace-path')).toContainText('no computational path');
    await page.getByRole('button', { name: 'Next', exact: true }).click();
    await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
    await page.getByRole('button', { name: 'Return to trace start', exact: true }).click();
    await expect(page.getByTestId('trace-step')).toContainText('Step 1 of');
    expect(await calls(page, 'TRACE')).toBe(1);
    expect(await calls(page, 'COMPARE')).toBe(comparisons);
    if (id === 'effect') {
      await page.getByRole('button', { name: 'Exit trace', exact: true }).click();
      await page.locator('.guided-experiments').evaluate(element => element.scrollIntoView({ block: 'start' }));
      await page.screenshot({ path: testInfo.outputPath('examples-desktop.png') });
    }
  }
});

test('drafts retain the completed guide; pending edits, replay, restart and checkpoint changes cannot falsely label a recipe', async ({ page }) => {
  await record(page);
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `{
      const send = self.postMessage.bind(self), delayed = new Set();
      self.addEventListener('message', event => { if (event.data?.command === 'COMPARE') delayed.add(event.data.id); });
      self.postMessage = (message, ...args) => delayed.has(message?.id)
        ? setTimeout(() => { delayed.delete(message.id); send(message, ...args); }, 500) : send(message, ...args);
    }\n${await response.text()}` });
  });
  await page.goto('?backend=cpu&view=details');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const comparisons = await calls(page, 'COMPARE');
  await page.getByLabel('Binding 1 value').selectOption('2');
  await expect(guide(page)).toContainText('Draft choices have not been applied');
  await page.getByRole('button', { name: 'Inspect Layer 1 Layer input', exact: true }).click();
  await page.getByRole('button', { name: 'Choose source and position', exact: true }).click();
  expect(await calls(page, 'COMPARE')).toBe(comparisons);
  await expect(guide(page).getByRole('heading')).toHaveText(titles.effect);
  await expect(page.locator('.examples-note')).toContainText('replaces your input or source draft');
  await page.getByRole('button', { name: `Start example: ${titles.unreachable}`, exact: true }).click();
  await expect(page.getByRole('button', { name: 'Trace this example', exact: true })).toBeEnabled();
  await expect(guide(page).getByRole('heading')).toHaveText(titles.unreachable);
  await expect(page.getByLabel('Binding 1 value')).toHaveValue('0');
  await expect(page.locator('.residual-copy')).toHaveCount(0);
  await page.getByRole('slider', { name: 'Head strength', exact: true }).fill('0.37');
  await expect(guide(page)).toContainText('Last completed example');
  await expect(guide(page).getByRole('heading')).toHaveText(titles.unreachable);
  await expect(page.getByRole('button', { name: 'Trace this example', exact: true })).toBeDisabled();
  await expect(page.getByTestId('recipe-note')).toContainText(titles.unreachable);
  await expect(page.locator('.status')).toHaveText('Showing 0.37×');
  await expect(guide(page)).toContainText('Custom experiment');
  await expect(page.locator('.example-card[aria-pressed=true]')).toHaveCount(0);
  await expect(page.getByTestId('recipe-note')).toHaveCount(0);
  await page.getByRole('button', { name: 'Start example: Ablation changes the answer', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Trace this example', exact: true })).toBeEnabled();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = await readFile((await (await download).path())!, 'utf8');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(guide(page)).toContainText('Custom experiment');
  await page.getByLabel('Import experiment').setInputFiles({ name: 'example.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches');
  await expect(guide(page)).toContainText('Custom experiment');
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await expect(guide(page)).toContainText('Custom experiment');
  await page.getByLabel('Model', { exact: true }).selectOption('verification');
  await expect(page.locator('.status')).toHaveText('Showing 0.50×');
  await expect(page.locator('.guided-experiments')).toHaveCount(0);
  await page.getByLabel('Model', { exact: true }).selectOption('retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await expect(guide(page).getByRole('heading')).toHaveText(titles.effect);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('example cards and guide actions work by keyboard at mobile and tablet widths', async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const width of [360, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('?backend=cpu&view=details');
    await expect(page.locator('.status')).toHaveText('Showing 0.00×');
    if (width === 390) {
      await page.locator('.guided-experiments').evaluate(element => element.scrollIntoView({ block: 'start' }));
      await page.screenshot({ path: testInfo.outputPath('examples-mobile.png') });
    }
    const card = page.getByRole('button', { name: `Start example: ${titles.small}`, exact: true });
    await card.focus(); await page.keyboard.press('Enter');
    await expect(guide(page).getByRole('heading')).toHaveText(titles.small);
    const inspect = page.getByRole('button', { name: 'Inspect the edited head', exact: true });
    await expect(inspect).toBeEnabled(); await inspect.focus(); await page.keyboard.press('Enter');
    await expect(page.locator('.node-panel-heading h2')).toHaveText('Layer 1 · Head 2');
    await page.getByRole('button', { name: 'Trace this example', exact: true }).focus(); await page.keyboard.press('Enter');
    await expect(page.getByTestId('trace-step')).toContainText('Step 1 of');
    await page.getByRole('button', { name: 'Next', exact: true }).focus(); await page.keyboard.press('Enter');
    await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
    for (const selector of ['.trace-navigation', '.prediction-summary']) expect(await page.locator(selector).evaluate(element => {
      const box = element.getBoundingClientRect(); return box.left >= -1 && box.right <= innerWidth + 1 && box.top >= 0 && box.bottom <= innerHeight;
    })).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    expect(await page.locator('.example-cards').evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  }
});
