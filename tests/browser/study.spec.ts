import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import type { SessionDocument } from '../../src/study/session';

async function setUp(page: Page, id: string) {
  await page.getByLabel('Participant ID').fill(id);
  await page.getByLabel('Experience band').selectOption('Took an ML course');
  await page.getByLabel('Device and browser').fill(`Playwright ${test.info().project.name}`);
  await page.getByLabel('The participant gave informed consent').check();
  await page.getByRole('button', { name: 'Start session' }).click();
}
async function predict(page: Page, direction: string, magnitude: string, attention: string) {
  await page.getByRole('radiogroup', { name: 'P1 direction' }).getByLabel(direction, { exact: true }).check();
  await page.getByRole('radiogroup', { name: 'P2 magnitude' }).getByLabel(magnitude).check();
  await page.getByLabel('P3 reasoning').fill(`Because ${direction}`);
  await page.getByRole('radiogroup', { name: 'P4 attention' }).getByLabel(attention, { exact: true }).check();
}

test('a facilitated session records predictions before revealing each live recipe result', async ({ page }) => {
  test.setTimeout(120_000);
  const requests: string[] = [], errors: string[] = [];
  page.on('request', request => requests.push(request.url()));
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×', { timeout: 30_000 });
  expect(requests.filter(url => /study/u.test(url))).toEqual([]);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('study.html');
  await expect(page.locator('iframe')).toHaveCount(0);
  await page.getByLabel('Participant ID').fill('Alice');
  await page.getByLabel('Experience band').selectOption('Took an ML course');
  await page.getByLabel('The participant gave informed consent').check();
  await expect(page.getByRole('button', { name: 'Start session' })).toBeDisabled();
  await page.getByLabel('The participant gave informed consent').uncheck();
  await page.getByLabel('Participant ID').fill('p01');
  await expect(page.getByRole('button', { name: 'Start session' })).toBeDisabled();
  await setUp(page, 'p01');
  await expect(page.getByRole('blockquote')).toContainText('We are testing the tool, not you.');
  await page.getByRole('button', { name: 'Continue to orientation' }).click();
  // Orientation shows a neutral baseline, never Task 1's recipe or its result.
  const app = page.frameLocator('iframe');
  await expect(page.locator('.frame-host:not(.hidden) iframe')).toBeVisible({ timeout: 30_000 });
  await expect(app.locator('.status')).toHaveText('Showing 1.00×');
  await expect(app.locator('#sequence')).toHaveValue('A = 1 ; B = 2 ; C = 3 ; ? C =');
  await expect(app.getByTestId('recipe-note')).toHaveCount(0);
  await page.getByLabel('Orientation notes').fill('Found the slider quickly.');
  await page.getByRole('button', { name: 'Continue to Task 1' }).click();

  const plan = [
    { recipe: 'effect', predicted: ['Down', 'Large (over 50 points)', 'No'], observed: /3 99\.99% → 0\.00% · down, large$/u, next: 'Continue to Task 2' },
    { recipe: 'small', predicted: ['Stay the same', 'No change', 'Yes'], observed: /2 99\.99% → 99\.99% · down, tiny$/u, next: 'Continue to Task 3' },
    { recipe: 'unreachable', predicted: ['Stay the same', 'No change', 'No'], observed: /3 99\.99% → 99\.99% · same, none · no computational path$/u, next: 'Continue to concept probes' },
  ];
  for (const step of plan) {
    // Nothing about the result is on screen until the prediction is locked.
    await expect(page.locator('iframe')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Record prediction and show the result' })).toBeDisabled();
    await predict(page, ...step.predicted as [string, string, string]);
    await page.getByRole('button', { name: 'Record prediction and show the result' }).click();
    await expect(page.getByTestId('observed')).toHaveText(step.observed, { timeout: 30_000 });
    await expect(page.frameLocator('iframe').getByLabel('Prepared experiment')).toHaveValue(step.recipe);
    await expect(page.getByRole('radiogroup', { name: 'P1 direction' }).getByRole('radio').first()).toBeDisabled();
    await page.getByLabel('R1 match').fill(`Observed for ${step.recipe}`);
    await page.getByRole('button', { name: step.next }).click();
  }
  for (const probe of ['C1', 'C2', 'C3', 'C4', 'C5']) await page.getByLabel(`${probe} answer`).fill(`Answer ${probe}`);
  await page.getByLabel('C3 hint given').check();
  await page.getByRole('button', { name: 'Continue to debrief' }).click();
  await page.getByLabel('Debrief notes').fill('Wanted a label on the path note.');
  await page.getByRole('button', { name: 'Finish and export' }).click();
  await expect(page.getByRole('heading', { name: 'Session P01 recorded' })).toBeVisible();
  await expect(page.locator('tbody tr')).toHaveCount(3);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download session JSON' }).click();
  const session = JSON.parse(await readFile((await (await download).path())!, 'utf8')) as SessionDocument;
  expect(session.participant).toEqual({ id: 'P01', experienceBand: 'Took an ML course', device: `Playwright ${test.info().project.name}` });
  expect(session.release.commit).toMatch(/^[0-9a-f]{40}$/u);
  expect(session.tasks.map(task => [task.recipeId, task.observation.recipeTitle])).toEqual([
    ['effect', 'Ablation changes the answer'], ['small', 'A reachable edit has little effect'], ['unreachable', 'An earlier final-layer edit cannot reach the answer']]);
  for (const task of session.tasks) expect(Date.parse(task.prediction.recordedAt)).toBeLessThanOrEqual(Date.parse(task.observation.revealedAt));
  expect(session.tasks[1].observation.logProbabilityDeltaText).toBe('-0.0000');
  expect(session.probes.C3).toEqual({ answer: 'Answer C3', hint: true });
  expect(JSON.stringify(session)).not.toContain('Alice');
  expect(errors).toEqual([]);
});

test('an unfinished session can be resumed after a reload, then discarded', async ({ page }) => {
  await page.goto('study.html');
  await setUp(page, 'P02');
  await page.getByRole('button', { name: 'Continue to orientation' }).click();
  await page.getByRole('button', { name: 'Continue to Task 1' }).click();
  await predict(page, 'Up', 'Moderate', 'Unsure');
  await page.reload();
  await page.getByRole('button', { name: 'Resume P02' }).click();
  await expect(page.getByRole('heading', { name: 'Task 1: prediction card' })).toBeVisible();
  await expect(page.getByRole('radiogroup', { name: 'P2 magnitude' }).getByLabel('Moderate')).toBeChecked();
  await expect(page.locator('iframe')).toHaveCount(0);
  await page.reload();
  await page.getByRole('button', { name: 'Discard' }).click();
  await expect(page.getByLabel('Participant ID')).toHaveValue('');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Resume P02' })).toHaveCount(0);
});
