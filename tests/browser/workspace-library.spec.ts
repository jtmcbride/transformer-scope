import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import fixture from '../fixtures/story-trajectory-workspace.json' with { type: 'json' };
import { workspaceLibraryKey, workspaceStorageKey } from '../../src/story-workspace-library';
import { loadShippedArtifact } from '../engine/retrieval-artifact';

const { manifest } = loadShippedArtifact('tinystories');
const libraryKey = workspaceLibraryKey(manifest), legacyKey = workspaceStorageKey(manifest);
async function ready(page: Page) {
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
}
async function upload(page: Page, data: unknown) {
  await page.getByLabel('Open story workspace').setInputFiles({ name: 'workspace.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(data)) });
}
async function download(page: Page, name = 'Download workspace') {
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name, exact: true }).click();
  return JSON.parse(await readFile((await (await pending).path())!, 'utf8'));
}
async function rename(page: Page, kind: 'workspace' | 'comparison', value: string) {
  await page.getByRole('button', { name: `Rename ${kind}`, exact: true }).click();
  await page.getByRole('textbox', { name: kind === 'workspace' ? 'Workspace name' : 'Comparison name', exact: true }).fill(value);
  if (kind === 'comparison') await page.getByRole('textbox', { name: 'Comparison name', exact: true }).press('Enter');
  else await page.getByRole('button', { name: 'Save name', exact: true }).click();
  await expect(page.getByRole('button', { name: `Rename ${kind}`, exact: true })).toBeFocused();
}
async function library(page: Page) {
  const details = page.locator('.workspace-library');
  if (await details.getAttribute('open') === null) await details.locator(':scope > summary').click();
  return details;
}

test('names, nested ancestry and shared paths survive new prompts, reload and file round trips', async ({ page }, info) => {
  test.setTimeout(180_000);
  await ready(page); await upload(page, fixture.workspace);
  await rename(page, 'workspace', 'The red ball experiment');
  const map = page.getByRole('navigation', { name: 'Story branch map' });
  const child = fixture.workspace.nodes[1];
  await expect(map.locator(`li[data-history-id="${child.parentId}"] > ul > li[data-history-id="${child.id}"]`)).toHaveCount(1);
  await expect(map.getByRole('button', { name: /^Comparison 1:/ })).toContainText('Muted');
  await map.getByRole('button', { name: /^Comparison 2:/ }).click();
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await rename(page, 'comparison', 'Ask a friend');
  await expect(map.getByRole('button', { name: 'Comparison 2: Ask a friend', exact: true })).toHaveAttribute('aria-current', 'location');
  await expect(map.locator('[aria-current="location"]')).toHaveCount(1);
  await page.locator('.story-map-comparison > summary').click();
  await expect(page.getByLabel('Story A path').locator('option')).toContainText(['Ask a friend · Shared 1 A · Keep source settings']);
  const named = await download(page);
  expect(named.version).toBe(3); expect(named.root).toEqual(fixture.workspace.root);
  expect(named.sharedForks).toEqual(fixture.workspace.sharedForks);
  expect(named.nodes[1].name).toBe('Ask a friend');
  expect(named.nodes.map(({ name: _name, ...node }: any) => node)).toEqual(fixture.workspace.nodes);
  await rename(page, 'comparison', '');
  expect((await download(page)).nodes[1]).not.toHaveProperty('name');
  await rename(page, 'comparison', 'Ask a friend');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: /^Branch history/ }).click();
  await page.getByRole('region', { name: 'Story branch workspace', exact: true }).screenshot({ path: info.outputPath('named-ancestry-mobile.png') });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByLabel('Story prompt').fill('Lily saw a little dog.');
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Ready to continue the completed prompt.', { timeout: 60_000 });
  const saved = await library(page);
  await expect(saved.getByRole('button', { name: 'Open workspace: The red ball experiment', exact: true })).toBeVisible();
  await saved.screenshot({ path: info.outputPath('saved-workspaces-desktop.png') });
  await page.reload(); await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await (await library(page)).getByRole('button', { name: 'Open workspace: The red ball experiment', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  expect(await download(page)).toEqual(named);
  await page.getByRole('button', { name: /^Inspect with head change token/ }).first().click();
  await expect(page.getByLabel('Active story inspection')).toContainText('Ask a friend');
  const imported = { ...named, name: 'Imported experiment' };
  await upload(page, imported);
  expect(await download(page)).toEqual(imported);
  await expect((await library(page)).getByRole('button', { name: 'Open workspace: The red ball experiment', exact: true })).toBeVisible();
  await page.getByRole('region', { name: 'Story branch workspace', exact: true }).screenshot({ path: info.outputPath('named-ancestry-desktop.png') });
});

test('resetting and switching while a story is playing preserve completed work without late writes', async ({ page }) => {
  test.setTimeout(120_000);
  await ready(page);
  await page.getByLabel('New-token limit').fill('12');
  await page.getByRole('button', { name: 'Play story', exact: true }).click();
  await expect(page.getByRole('button', { name: /^Inspect original token 1:/ })).toBeVisible({ timeout: 60_000 });
  await page.getByRole('button', { name: 'Reset story', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Ready to continue the completed prompt.');
  const saved = await library(page), entry = saved.locator('li[data-workspace-id]').first();
  const oldId = await entry.getAttribute('data-workspace-id');
  const captured = await page.evaluate(({ key, id }) => JSON.parse(localStorage.getItem(key)!).entries.find((entry: any) => entry.id === id).data, { key: libraryKey, id: oldId });
  expect(captured.root.frame.step).toBeGreaterThanOrEqual(1);
  await page.getByLabel('New-token limit').fill('2');
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  const current = await download(page);
  await entry.getByRole('button', { name: /^Open workspace:/ }).click();
  await expect(page.locator('.continuation-status')).toContainText(`Paused after ${captured.root.frame.step}`);
  expect(await download(page)).toEqual(captured);
  const rows = (await library(page)).locator('li[data-workspace-id]');
  await rows.first().getByRole('button', { name: /^Open workspace:/ }).click();
  expect(await download(page)).toEqual(current);
  const archived = await page.evaluate(({ key, id }) => JSON.parse(localStorage.getItem(key)!).entries.find((entry: any) => entry.id === id).data, { key: libraryKey, id: oldId });
  expect(archived).toEqual(captured);
});

test('failed saves block replacement and removal requires a named confirmation', async ({ page }) => {
  test.setTimeout(120_000);
  await page.addInitScript(() => {
    const set = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, value) {
      if ((window as any).blockStorySaving && key.startsWith('transformer-story-workspace:')) throw new DOMException('Quota exceeded', 'QuotaExceededError');
      return set.call(this, key, value);
    };
  });
  await ready(page); await upload(page, { ...fixture.workspace, name: 'Keep this story' });
  const before = await download(page);
  await page.evaluate(() => { (window as any).blockStorySaving = true; });
  await page.getByRole('button', { name: 'Reset story', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Automatic saving is unavailable');
  expect(await download(page)).toEqual(before);
  await page.getByLabel('Story prompt').fill('Once upon a time, a bird sang.');
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.workspace-pending-prompt')).toBeVisible({ timeout: 60_000 });
  expect(await download(page)).toEqual(before);
  await page.evaluate(() => { (window as any).blockStorySaving = false; });
  await page.getByRole('button', { name: 'Start new prompt workspace', exact: true }).click();
  await expect(page.locator('.continuation-status')).toHaveText('Ready to continue the completed prompt.');
  const saved = await library(page);
  await saved.getByRole('button', { name: 'Remove workspace: Keep this story', exact: true }).click();
  const confirmation = saved.getByRole('group', { name: 'Remove saved workspace: Keep this story', exact: true });
  await confirmation.getByRole('button', { name: 'Keep workspace', exact: true }).click();
  await expect(saved.getByRole('button', { name: 'Open workspace: Keep this story', exact: true })).toBeVisible();
  await saved.getByRole('button', { name: 'Remove workspace: Keep this story', exact: true }).click();
  await confirmation.getByRole('button', { name: 'Remove saved workspace', exact: true }).click();
  await expect(saved.getByRole('button', { name: 'Open workspace: Keep this story', exact: true })).toHaveCount(0);
  await expect(saved.locator(':scope > summary')).toBeFocused();
});

test('legacy saved work migrates and unreadable library contents remain downloadable', async ({ page }) => {
  test.setTimeout(120_000);
  const old = { ...fixture.workspace, name: 'Legacy experiment' };
  await page.addInitScript(({ key, data }) => { if (!localStorage.getItem(key)) localStorage.setItem(key, data); }, { key: legacyKey, data: JSON.stringify(old) });
  await ready(page); expect(await download(page)).toEqual(old);
  expect(await page.evaluate(key => localStorage.getItem(key), legacyKey)).toBe(JSON.stringify(old));
  await page.evaluate(key => localStorage.setItem(key, '{unreadable library'), libraryKey);
  await page.reload(); await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await expect(page.getByRole('alert')).toContainText('original contents');
  const saved = await library(page);
  await expect(saved.getByRole('button', { name: 'Open workspace: Legacy experiment', exact: true })).toBeEnabled();
  await expect(saved.getByRole('button', { name: 'Open workspace: Recovered saved data', exact: true })).toBeDisabled();
  const pending = page.waitForEvent('download');
  await saved.getByRole('button', { name: 'Download workspace: Recovered saved data', exact: true }).click();
  expect(await readFile((await (await pending).path())!, 'utf8')).toBe('{unreadable library');
  await page.getByRole('button', { name: 'Step one token', exact: true }).click();
  await expect(page.locator('.continuation-status')).toContainText('Paused after 1 step', { timeout: 60_000 });
  await page.reload();
  await expect((await library(page)).getByRole('button', { name: 'Download workspace: Recovered saved data', exact: true })).toBeVisible();
});

test('another tab cannot silently replace this tab’s unsaved work or erase its library changes', async ({ page, context }) => {
  test.setTimeout(120_000);
  await ready(page); await upload(page, { ...fixture.workspace, name: 'Shared browser story' });
  const other = await context.newPage(); await ready(other);
  await rename(other, 'workspace', 'Name from another tab');
  await expect(page.getByRole('alert')).toContainText('another tab');
  await rename(page, 'workspace', 'My unsaved name');
  await expect(page.getByLabel('Current story workspace')).toContainText('Not saved yet');
  expect((await download(page)).name).toBe('My unsaved name');
  const saved = await page.evaluate(key => JSON.parse(localStorage.getItem(key)!), libraryKey);
  expect(saved.entries.find((entry: any) => entry.id === saved.activeId).data.name).toBe('Name from another tab');
  await page.reload();
  await expect(page.getByLabel('Current story workspace')).toContainText('Name from another tab', { timeout: 60_000 });
  await other.close();
});

test('renaming keeps measured interactions and a prepared paired branch usable', async ({ page }) => {
  test.setTimeout(120_000);
  await ready(page); await upload(page, fixture.workspace);
  await page.getByRole('button', { name: /^Comparison 2:/ }).click();
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced', { timeout: 60_000 });
  await page.locator('.story-map-comparison > summary').click();
  const shared = page.getByRole('region', { name: 'Shared story continuations', exact: true });
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');
  const trace = shared.getByRole('region', { name: 'Head-pair interaction along the stories', exact: true });
  await trace.getByLabel('Interaction trajectory target mode').selectOption('fixed');
  await trace.getByLabel('Chosen next token').selectOption('471');
  await trace.getByRole('button', { name: 'Measure token-by-token interactions', exact: true }).click();
  await expect(trace.getByTestId('trajectory-status')).toContainText('Token interactions complete.', { timeout: 60_000 });
  const measured = await download(page, 'Download token interactions');
  await trace.getByRole('button', { name: 'Set up both branches from offset 1', exact: true }).click();
  const setup = shared.getByRole('group', { name: 'Rebranch both stories from trajectory', exact: true });
  await setup.getByLabel('Rebranch first head strength').selectOption('2');
  await rename(page, 'comparison', 'A different direction');
  await rename(page, 'workspace', 'Compare friendly paths');
  await expect(setup).toBeVisible();
  await expect(setup.getByLabel('Rebranch first head strength')).toHaveValue('2');
  await expect(trace.getByTestId('trajectory-status')).toContainText('Token interactions complete.');
  expect((await download(page, 'Download token interactions')).rows).toEqual(measured.rows);
  await expect(shared.getByRole('article', { name: 'Story A continuation comparison', exact: true })).toContainText('A different direction');
  const saved = await download(page);
  expect(saved.root).toEqual(fixture.workspace.root);
  expect(saved.nodes[1].frame).toEqual(fixture.workspace.nodes[1].frame);
});
