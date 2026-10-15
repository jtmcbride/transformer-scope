import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import continuation from '../fixtures/continuation-reference.json' with { type: 'json' };
import reference from '../fixtures/story-forks-reference.json' with { type: 'json' };

test.use({ actionTimeout: 20_000 });

async function record(page: Page, name = 'Download branch comparison') {
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name, exact: true }).click();
  return JSON.parse(await readFile((await (await pending).path())!, 'utf8'));
}
async function ready(page: Page, edited = false) {
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByLabel('Story prompt').fill(continuation.prompt);
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  if (edited) {
    await page.getByRole('button', { name: 'Mute selected head', exact: true }).click();
    await expect(page.locator('.status')).toHaveText('Showing 0.00×');
    await page.getByLabel('Compare with current head edit').check();
  }
  await page.getByLabel('New-token limit').fill('8');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
}
function matches(saved: any, example: typeof reference.cases[number]) {
  expect(saved.format).toBe('transformer-microscope-story-fork');
  expect(saved.checkpoint.modelRevision).toBe(reference.modelRevision);
  expect(saved.plan.source.prefix).toEqual(example.plan.source.prefix);
  expect(saved.plan.source.step).toBe(example.plan.source.step);
  expect(saved.drawOffset).toBe(example.plan.source.step - 1);
  expect(saved.plan.edit).toEqual(example.plan.edit);
  expect(saved.plan.duration).toBe(example.plan.duration);
  for (const branch of ['original', 'edited'] as const) {
    expect(saved.frame[branch].tokenIds).toEqual(example[branch].tokenIds);
    saved.frame[branch].choices.forEach((choice: { probability: number }, i: number) => expect(choice.probability).toBeCloseTo(example[branch].choices[i].probability, 5));
  }
}

test('forking an original token preserves its story and matches independent sustained and single-prediction edits', async ({ page }, testInfo) => {
  test.setTimeout(150_000);
  await ready(page);
  const source = await record(page, 'Download continuation');
  await page.getByRole('button', { name: /^Inspect original token 6:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await inspection.getByRole('button', { name: 'L1 · Head 1', exact: true }).click();
  const fork = page.getByRole('region', { name: 'Branch from selected token', exact: true });
  await fork.getByLabel('Branch new-token limit').fill('12');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 90_000 });
  const sustained = await record(page);
  matches(sustained, reference.cases[0]);
  await page.getByRole('button', { name: /^Inspect original token 6:/ }).click();
  expect((await record(page)).frame).toEqual(sustained.frame);
  expect(sustained.firstDifferenceStep).toBe(6);
  await expect(fork.locator('.fork-difference')).toContainText('First different token at story step 6');
  await expect(fork.getByRole('article', { name: 'Keep source settings continuation' })).toContainText('He ran to his friend');
  await expect(fork.getByRole('article', { name: 'With head change continuation' })).toContainText('He asked his friend');
  expect((await record(page, 'Download continuation')).frame).toEqual(source.frame);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await fork.screenshot({ path: testInfo.outputPath('fork-comparison.png') });
  await fork.getByLabel('Branch edit duration').selectOption('prediction');
  await fork.getByRole('button', { name: 'Start new branch comparison', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 90_000 });
  const once = await record(page);
  matches(once, reference.cases[1]);
  expect(once.frame.original).toEqual(sustained.frame.original);
  expect(once.frame.edited.tokenIds).not.toEqual(sustained.frame.edited.tokenIds);
  await expect(page.locator('.status')).toHaveText('Showing 1.00×');
});

test('forking an edited token retains a different captured head, replaces the same head, and fits a narrow screen', async ({ page }) => {
  test.setTimeout(150_000);
  await ready(page, true);
  await page.getByRole('button', { name: /^Inspect edited token 7:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await inspection.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await inspection.getByRole('button', { name: 'L2 · Head 2', exact: true }).click();
  const fork = page.getByRole('region', { name: 'Branch from selected token', exact: true });
  await fork.getByLabel('Branch new-token limit').fill('12');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 90_000 });
  matches(await record(page), reference.cases[2]);
  await inspection.getByRole('button', { name: 'Previous model layer', exact: true }).click();
  await inspection.getByRole('button', { name: 'L1 · Head 1', exact: true }).click();
  await fork.getByRole('button', { name: 'Normal branch head', exact: true }).click();
  // Changing the draft must not relabel a finished comparison.
  await expect(fork.getByTestId('fork-captured-edit')).toContainText('Layer 2 · Head 2 · 0.00×');
  await fork.getByRole('button', { name: 'Start new branch comparison', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 90_000 });
  matches(await record(page), reference.cases[3]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await fork.evaluate(node => node.scrollWidth <= node.clientWidth + 1 && node.getBoundingClientRect().right <= innerWidth)).toBe(true);
  await fork.getByRole('button', { name: 'Clear comparison', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveCount(0);
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.');
});

test('a sampled identity fork keeps the original draw counter through cancellation and resume; navigation cancels unfinished work', async ({ page }) => {
  test.setTimeout(150_000);
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `${await response.text()}\n{
      const dispatch = self.onmessage;
      self.onmessage = event => {
        if (event.data?.command === 'BASELINE') {
          console.info('delayed-fork-baseline');
          setTimeout(() => dispatch.call(self, event), 800);
        } else dispatch.call(self, event);
      };
    }` });
  });
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByLabel('Continuation sampling').selectOption('sample');
  await page.getByLabel('New-token limit').fill('5');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.', { timeout: 60_000 });
  const source = await record(page, 'Download continuation');
  await page.getByRole('button', { name: /^Inspect original token 3:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  const fork = page.getByRole('region', { name: 'Branch from selected token', exact: true });
  await fork.getByRole('button', { name: 'Normal branch head', exact: true }).click();
  await fork.getByLabel('Branch new-token limit').fill('3');
  let inFlight = page.waitForEvent('console', message => message.text() === 'delayed-fork-baseline');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await inFlight;
  await fork.getByRole('button', { name: 'Stop comparison', exact: true }).click();
  await page.waitForTimeout(1000);
  await expect(fork.locator('.fork-tokens > button')).toHaveCount(0);
  await fork.getByRole('button', { name: 'Step comparison', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Comparison paused after 1 step.', { timeout: 60_000 });
  await fork.getByRole('button', { name: 'Play comparison', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60_000 });
  const saved = await record(page);
  expect(saved.frame.original.choices).toEqual(source.frame.original.choices.slice(2));
  expect(saved.frame.edited).toEqual(saved.frame.original);
  expect(saved.firstDifferenceStep).toBeNull();
  expect((await record(page, 'Download continuation')).frame).toEqual(source.frame);
  await fork.getByRole('button', { name: /^Inspect with head change token 3:/ }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspection.getByTestId('selected-token-probability')).toHaveText(`${(saved.frame.edited.choices[0].probability * 100).toFixed(4)}%`);
  await fork.getByRole('button', { name: 'Normal branch head', exact: true }).click();
  inFlight = page.waitForEvent('console', message => message.text() === 'delayed-fork-baseline');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await inFlight;
  await inspection.getByRole('button', { name: 'Back to previous comparison', exact: true }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await page.waitForTimeout(1000);
  expect((await record(page)).frame).toEqual(saved.frame);
  await expect(page.getByRole('alert')).toHaveCount(0);
  inFlight = page.waitForEvent('console', message => message.text() === 'delayed-fork-baseline');
  await fork.getByRole('button', { name: 'Start new branch comparison', exact: true }).click();
  await inFlight;
  await inspection.getByRole('button', { name: 'Next token', exact: false }).click();
  await expect(inspection).toContainText('Original story · step 4');
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(fork.locator('.fork-status')).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('either branch opens a replayed flow and supports further head changes with preserved ancestors', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    const live = new Set<Worker>();
    Object.assign(window, { liveStoryWorkers: live });
    window.Worker = class extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) { super(url, options); live.add(this); }
      terminate() { live.delete(this); super.terminate(); }
    };
  });
  const workerCount = () => page.evaluate(() => (window as unknown as { liveStoryWorkers: Set<Worker> }).liveStoryWorkers.size);
  const nested = (await import('../fixtures/nested-story-forks-reference.json', { with: { type: 'json' } })).default;
  await ready(page);
  await page.getByRole('button', { name: /^Inspect original token 6:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  const fork = page.getByRole('region', { name: 'Branch from selected token', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await fork.getByLabel('Branch new-token limit').fill('8');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60_000 });
  const parent = await record(page);
  await fork.getByRole('button', { name: /^Inspect keep source settings token 6:/ }).focus();
  await page.keyboard.press('Enter');
  await expect(inspection).toHaveCount(1);
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspection).toContainText('Comparison 1 · Keep source settings story · step 6');
  await expect.poll(workerCount).toBe(2);
  await expect(inspection.getByRole('button', { name: 'Previous token', exact: false })).toBeDisabled();
  await expect(inspection.getByTestId('selected-token-probability')).toHaveText(`${(parent.frame.original.choices[0].probability * 100).toFixed(4)}%`);
  await inspection.getByRole('button', { name: 'Next token', exact: false }).click();
  await expect(inspection).toContainText('Keep source settings story · step 7');
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await inspection.getByRole('button', { name: 'Back to previous comparison', exact: true }).click();
  await expect(fork.getByRole('button', { name: /^Inspect keep source settings token 7:/ })).toBeFocused();
  expect((await record(page)).frame).toEqual(parent.frame);

  await fork.getByRole('button', { name: /^Inspect with head change token 7:/ }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspection).toContainText('Captured edit: Layer 1 · Head 1 · 0.00×');
  await expect(inspection.getByTestId('selected-token-probability')).toHaveText(`${(parent.frame.edited.choices[1].probability * 100).toFixed(4)}%`);
  await inspection.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await inspection.getByRole('button', { name: 'L2 · Head 2', exact: true }).click();
  await fork.getByLabel('Branch new-token limit').fill('6');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60_000 });
  const second = await record(page);
  matches(second, nested.cases[0] as unknown as typeof reference.cases[number]);
  expect(second.schemaVersion).toBe(2);
  expect(second.plan.source.edits).toEqual([parent.plan.edit]);

  await fork.getByRole('button', { name: /^Inspect with head change token 9:/ }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspection).toContainText('Comparison 2 · With head change story · step 9');
  await expect.poll(workerCount).toBe(2);
  await expect(inspection).toContainText('Captured edits: Layer 1 · Head 1 · 0.00×; Layer 2 · Head 2 · 0.00×');
  await expect(inspection.getByRole('button', { name: 'L2 · Head 2', exact: true })).toContainText('Captured edit · 0.00×');
  await inspection.getByRole('button', { name: 'Previous model layer', exact: true }).click();
  await expect(inspection.getByRole('button', { name: 'L1 · Head 1', exact: true })).toContainText('Captured edit · 0.00×');
  await inspection.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await inspection.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await inspection.getByRole('button', { name: 'L3 · Head 3', exact: true }).click();
  await fork.getByLabel('Branch new-token limit').fill('6');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60_000 });
  matches(await record(page), nested.cases[2] as unknown as typeof reference.cases[number]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await inspection.evaluate(node => node.scrollWidth <= node.clientWidth + 1 && node.getBoundingClientRect().right <= innerWidth)).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await fork.screenshot({ path: testInfo.outputPath('nested-comparison.png') });
  await inspection.getByRole('button', { name: 'Back to previous comparison', exact: true }).click();
  expect((await record(page)).frame).toEqual(second.frame);
  await inspection.getByRole('button', { name: 'Back to previous comparison', exact: true }).click();
  expect((await record(page)).frame).toEqual(parent.frame);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('inspecting a one-prediction branch preserves expiry when branching again and restores the parent', async ({ page }) => {
  test.setTimeout(180_000);
  const nested = (await import('../fixtures/nested-story-forks-reference.json', { with: { type: 'json' } })).default;
  await ready(page);
  await page.getByRole('button', { name: /^Inspect original token 6:/ }).click();
  const inspection = page.getByRole('region', { name: 'Generated token computation', exact: true });
  const fork = page.getByRole('region', { name: 'Branch from selected token', exact: true });
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await fork.getByLabel('Branch edit duration').selectOption('prediction');
  await fork.getByLabel('Branch new-token limit').fill('6');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60_000 });
  const parent = await record(page);
  await fork.getByRole('button', { name: /^Inspect with head change token 6:/ }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspection).toContainText('Captured edit: Layer 1 · Head 1 · 0.00×');
  await inspection.getByRole('button', { name: 'Next token', exact: false }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspection).toContainText('Original model with no intervention');
  await inspection.getByRole('button', { name: 'Previous token', exact: false }).click();
  await expect(inspection.getByTestId('selected-token-result')).toBeVisible({ timeout: 60_000 });
  await inspection.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await inspection.getByRole('button', { name: 'L2 · Head 2', exact: true }).click();
  await fork.getByLabel('Branch edit duration').selectOption('prediction');
  await fork.getByLabel('Branch new-token limit').fill('6');
  await fork.getByRole('button', { name: 'Compare from this token', exact: true }).click();
  await expect(fork.locator('.fork-status')).toHaveText('Branch comparison complete.', { timeout: 60_000 });
  const child = await record(page);
  matches(child, nested.cases[1] as unknown as typeof reference.cases[number]);
  expect(child.plan.source.continuationEdits).toEqual([]);
  await inspection.getByRole('button', { name: 'Back to previous comparison', exact: true }).click();
  expect((await record(page)).frame).toEqual(parent.frame);
});
