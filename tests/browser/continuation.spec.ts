import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import reference from '../fixtures/continuation-reference.json' with { type: 'json' };

async function ready(page: Page) {
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
}
async function download(page: Page) {
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download continuation', exact: true }).click();
  return JSON.parse(await readFile((await (await pending).path())!, 'utf8'));
}

test('stepping and paired playback match sixteen independent original/edited tokens with a frozen edit', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  await ready(page);
  await page.getByLabel('Story prompt').fill(reference.prompt);
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  await page.getByRole('button', { name: 'Mute selected head', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByLabel('Compare with current head edit').check();
  await page.getByLabel('New-token limit').fill('16');
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Paused after 1 step. Step or play to continue.', { timeout: 60_000 });
  await expect(page.getByRole('group', { name: 'Workspace view' })).toHaveCount(1);
  await expect(page.getByRole('article', { name: 'Original story' }).locator('.generated-tokens > button')).toHaveCount(1);
  await page.getByRole('button', { name: 'Restore original prediction', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
  await expect(page.getByRole('region', { name: 'Token by token story continuation' })).toContainText('Captured edit: Layer 1 · Head 1 · 0.00×');
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step');
  await page.getByRole('button', { name: 'Visual flow', exact: true }).click();
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 90_000 });
  const saved = await download(page);
  expect(saved.format).toBe('transformer-microscope-continuation');
  expect(saved.checkpoint.modelRevision).toBe(reference.modelRevision);
  expect(saved.prompt).toEqual({ text: reference.prompt, tokenIds: reference.promptTokenIds });
  expect(saved.edit).toMatchObject({ layer: 0, head: 0, factor: 0, tokens: { kind: 'all' } });
  for (const key of ['original', 'edited'] as const) {
    expect(saved.frame[key].tokenIds).toEqual(reference[key].tokenIds);
    expect(saved.frame[key].finish).toBe('token limit');
    saved.frame[key].choices.forEach((choice: { probability: number }, i: number) => expect(choice.probability).toBeCloseTo(reference[key].choices[i].probability, 5));
  }
  await expect(page.getByTestId('original-story-text')).toContainText('He ran to his friend');
  await expect(page.getByTestId('edited-story-text')).toContainText('He asked his friend');
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.locator('.story-continuation').screenshot({ path: testInfo.outputPath('controlled-continuation.png') });
  // Step seven follows different prefixes in the two branches. The chosen token is NOT an input.
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  for (const branch of ['original', 'edited'] as const) {
    await page.getByRole('button', { name: new RegExp(`^Inspect ${branch} token 7:`) }).click();
    await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
    expect(parseFloat(await inspection.getByTestId('selected-token-probability').innerText()) / 100).toBeCloseTo(reference[branch].choices[6].probability, 5);
    await expect(inspection.locator('.visual-token-stream > span')).toHaveCount(reference.promptTokenIds.length + 6);
    await expect(inspection.locator('.visual-token-stream > span').last()).toHaveText(branch === 'original' ? '␠ran' : '␠asked');
    await expect(inspection.getByLabel('Context before selected token')).toHaveText(`${reference.prompt} play with it. He ${branch === 'original' ? 'ran' : 'asked'}`);
    await expect(inspection.getByRole('button', { name: 'Mute selected head', exact: true })).toHaveCount(0);
  }
  await expect(inspection).toContainText('Captured edit: Layer 1 · Head 1 · 0.00×');
  await inspection.getByRole('button', { name: 'L1 · Head 1', exact: true }).click();
  const numbers = inspection.getByTestId('node-values');
  await expect(numbers.locator('dd').nth(1)).toHaveText('0.0000');
  expect(Number(await numbers.locator('dd').first().innerText())).toBeGreaterThan(0);
  await inspection.getByRole('button', { name: 'Inspect full vector', exact: true }).click();
  await expect(inspection.getByRole('table', { name: 'Selected node vector dimensions' }).locator('tbody tr')).toHaveCount(32);
  await inspection.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await expect(inspection.getByRole('button', { name: 'L2 · Head 1', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await inspection.getByRole('button', { name: 'Read the prediction', exact: true }).click();
  await expect(inspection.getByTestId('node-values')).toBeVisible();
  await inspection.screenshot({ path: testInfo.outputPath('selected-generated-token.png') });
  await inspection.getByRole('button', { name: 'Previous token', exact: false }).click();
  await expect(inspection.getByRole('heading', { level: 2 }).first()).toContainText('␠asked');
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await inspection.getByRole('button', { name: 'Close token inspection', exact: true }).click();
  expect((await download(page)).frame).toEqual(saved.frame);
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
});

test('seeded sampling resumes reproducibly, ignores drafts, resets on a completed prompt, and fits mobile', async ({ page }) => {
  test.setTimeout(120_000);
  await ready(page);
  await page.getByLabel('Continuation sampling').selectOption('sample');
  await page.getByLabel('New-token limit').fill('4');
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.getByLabel('Story prompt').fill('This unrun draft must not become the continuation prefix.');
  await expect(page.getByRole('region', { name: 'Token by token story continuation' })).toContainText('Your prompt has unrun edits');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  const first = await download(page);
  // Keyboard selection, sampled probability, responsive historical flow, and resume without changing history.
  const chip = page.getByRole('button', { name: /^Inspect original token 1:/ });
  await chip.focus(); await page.keyboard.press('Enter');
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await expect(inspection).toContainText('Chosen by seeded sampling');
  expect(parseFloat(await inspection.getByTestId('selected-token-probability').innerText()) / 100).toBeCloseTo(first.frame.original.choices[0].probability, 5);
  await expect(inspection.locator('.visual-token-stream > span')).toHaveCount(first.prompt.tokenIds.length);
  await page.setViewportSize({ width: 390, height: 844 });
  const inspectedLayout = await inspection.evaluate(node => ({ right: node.getBoundingClientRect().right, viewport: innerWidth }));
  expect(inspectedLayout.right).toBeLessThanOrEqual(inspectedLayout.viewport);
  await expect(page.locator('.story-continuation')).toBeVisible();
  await page.getByRole('button', { name: 'Reset story', exact: true }).click();
  await expect(inspection).toHaveCount(0);
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  const second = await download(page);
  expect(second.frame).toEqual(first.frame);
  expect(first.prompt.text).not.toContain('unrun draft');
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Ready to continue the completed prompt.');
  await expect(page.locator('.generated-tokens > button')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  const layout = await page.locator('.story-continuation').evaluate(node => ({ width: node.getBoundingClientRect().width, scroll: node.scrollWidth, viewport: innerWidth, right: node.getBoundingClientRect().right }));
  expect(layout.scroll).toBeLessThanOrEqual(layout.width + 1);
  expect(layout.right).toBeLessThanOrEqual(layout.viewport);
  // Digits are individual BPE pieces: 63 text tokens + BOS exactly fill this checkpoint.
  await page.getByLabel('Story prompt').fill('1'.repeat(63));
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await expect(page.getByRole('button', { name: 'Step one token', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.');
  await expect(page.getByRole('article', { name: 'Original story' })).toContainText('Stopped · context limit');
  await expect(page.getByRole('button', { name: 'Play story', exact: true })).toBeDisabled();
});

test('switching, closing and resetting an in-flight token inspection discard late responses without changing the story', async ({ page }) => {
  test.setTimeout(120_000);
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `${await response.text()}\n{
      const dispatch = self.onmessage;
      self.onmessage = event => {
        if (event.data?.command === 'BASELINE') {
          console.info('delayed-inspection-baseline');
          setTimeout(() => dispatch.call(self, event), 800);
        } else dispatch.call(self, event);
      };
    }` });
  });
  await ready(page);
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 2 steps', { timeout: 60_000 });
  const before = await download(page);
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  let inFlight = page.waitForEvent('console', message => message.text() === 'delayed-inspection-baseline');
  await page.getByRole('button', { name: /^Inspect original token 1:/ }).click();
  await inFlight;
  await inspection.getByRole('button', { name: 'Next token', exact: false }).click();
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await expect(inspection).toContainText('Original story · step 2');
  await expect(inspection.locator('.visual-token-stream > span')).toHaveCount(before.prompt.tokenIds.length + 1);
  inFlight = page.waitForEvent('console', message => message.text() === 'delayed-inspection-baseline');
  await inspection.getByRole('button', { name: 'Previous token', exact: false }).click();
  await inFlight;
  await inspection.getByRole('button', { name: 'Close token inspection', exact: true }).click();
  await page.waitForTimeout(1000);
  await expect(inspection).toHaveCount(0);
  expect((await download(page)).frame).toEqual(before.frame);
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await page.getByRole('button', { name: /^Inspect original token 1:/ }).click();
  await expect(page.getByRole('button', { name: 'Stop generation', exact: true })).toBeDisabled();
  await expect(inspection.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  const paused = await download(page);
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText(`Paused after ${paused.frame.step + 1} steps`, { timeout: 60_000 });
  await expect(inspection).toContainText('Original story · step 1');
  await expect(inspection.locator('.visual-token-stream > span')).toHaveCount(before.prompt.tokenIds.length);
  expect((await download(page)).frame.original.choices.slice(0, 2)).toEqual(before.frame.original.choices);
  await page.getByRole('button', { name: 'Reset story', exact: true }).click();
  await expect(inspection).toHaveCount(0);
  await expect(page.locator('.generated-tokens > button')).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  inFlight = page.waitForEvent('console', message => message.text() === 'delayed-inspection-baseline');
  await page.getByRole('button', { name: /^Inspect original token 1:/ }).click();
  await inFlight;
  await page.getByLabel('Model', { exact: true }).selectOption('retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×', { timeout: 60_000 });
  await expect(inspection).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('stop cancels an in-flight token and a model switch disposes continuation without late UI updates', async ({ page }) => {
  test.setTimeout(120_000);
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `${await response.text()}\n{
      const dispatch = self.onmessage;
      self.onmessage = event => {
        if (event.data?.command === 'BASELINE') {
          console.info('delayed-story-baseline');
          setTimeout(() => dispatch.call(self, event), 800);
        } else dispatch.call(self, event);
      };
    }` });
  });
  await ready(page);
  const inFlight = page.waitForEvent('console', message => message.text() === 'delayed-story-baseline');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await inFlight;
  await page.getByRole('button', { name: 'Stop generation', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 0 steps');
  // Observe beyond the deliberately delayed worker response, so a late append would fail.
  await page.waitForTimeout(1000);
  await expect(page.locator('.generated-tokens > button')).toHaveCount(0);
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await page.getByLabel('Model', { exact: true }).selectOption('retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×', { timeout: 60_000 });
  await expect(page.getByRole('region', { name: 'Token by token story continuation' })).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});
