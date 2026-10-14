import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import type { Commands, Command, WorkerResponse } from '../../src/workers/protocol';
import fixture from '../fixtures/text-inference.json' with { type: 'json' };

async function exported(page: Page) {
  if (await page.locator('.workspace-utilities').getAttribute('open') === null) await page.getByText('Save, replay & runtime settings', { exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  return JSON.parse(await readFile((await (await download).path())!, 'utf8'));
}
function parity(actual: number[], expected: number[]) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, i) => expect(Math.abs(actual[i] - value), `logit ${i}`).toBeLessThanOrEqual(2e-5 + 1e-5 * Math.abs(value)));
}

test('the default real-text model preserves input, compares heads, replays, and restarts on CPU', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('?backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await expect(page.getByLabel('Model', { exact: true })).toHaveValue('tinystories');
  await expect(page.getByLabel('Story prompt')).toHaveValue(fixture.cases[0].text);
  await expect(page.locator('.visual-answer strong')).toHaveText('she');
  await page.getByText('About this trained model', { exact: true }).click();
  await expect(page.locator('.text-model-about')).toContainText('gate failed (56.0%, required 80%)');
  for (let i = 0; i < 3; i++) await page.getByRole('button', { name: 'Next model layer', exact: true }).click();
  await expect(page.locator('.visual-layer-navigation')).toContainText('Layer 4 of 4');
  await expect(page.locator('.visual-answer strong')).toHaveText('she');
  await page.getByRole('button', { name: 'L4 · Head 3', exact: true }).click();
  await page.getByLabel('Selected head output strength').fill('1.5');
  await expect(page.locator('.status')).toHaveText('Showing 1.50×', { timeout: 60_000 });
  const changed = await exported(page);
  parity(changed.results.baselineLogits, fixture.cases[0].logits);
  parity(changed.results.modifiedLogits, fixture.cases[0].variants[1].logits);
  await page.getByLabel('Model', { exact: true }).selectOption('retrieval');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  // Runtime settings remain open across the model switch.
  await page.getByLabel('Import experiment').setInputFiles({ name: 'text-experiment.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(changed)) });
  await expect(page.getByLabel('Model', { exact: true })).toHaveValue('tinystories');
  await expect(page.locator('.status')).toHaveText('Showing 1.50×', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches saved results');
  await page.getByRole('button', { name: 'Visual flow', exact: true }).click();
  const text = fixture.cases[2].text;
  await page.getByLabel('Story prompt').fill(text);
  await page.getByRole('button', { name: 'Run input', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.50×', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Restore original prediction', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  const pendingDraft = 'This draft should not replace the completed input.';
  await page.getByLabel('Story prompt').fill(pendingDraft);
  if (await page.locator('.workspace-utilities').getAttribute('open') === null) await page.getByText('Save, replay & runtime settings', { exact: true }).click();
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await expect(page.getByLabel('Story prompt')).toHaveValue(text);
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Detailed', exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = JSON.parse(await readFile((await (await download).path())!, 'utf8'));
  expect(saved.input).toEqual({ text, tokenIds: fixture.cases[2].tokenIds });
  parity(saved.results.baselineLogits, fixture.cases[2].logits);
  await page.getByLabel('Story prompt').fill('Hello '.repeat(100));
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.getByRole('alert')).toContainText('shorten the passage');
  await expect(page.getByTestId('baseline-answer')).toHaveText('He');
  expect(errors).toEqual([]);
  await page.getByLabel('Story prompt').fill(fixture.cases[0].text);
  await page.getByRole('button', { name: 'Run input' }).click();
  await expect(page.locator('.status')).toHaveText('Showing 1.00×', { timeout: 60_000 });
  await page.getByRole('button', { name: 'Visual flow', exact: true }).click();
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.screenshot({ path: testInfo.outputPath('tinystories-text.png'), fullPage: true });
});

test('TinyStories CPU and supported WebGL match the independent forward, gradient, Adam, and intervention fixtures', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  let workerUrl = '';
  page.on('request', request => { if (/model\.worker-.*\.js/u.test(request.url())) workerUrl = request.url(); });
  await page.goto('?model=retrieval&backend=cpu');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const records = await page.evaluate(async ({ url, cases, revision }) => {
    const records = [];
    for (const backend of ['cpu', 'webgl'] as const) {
      const worker = new Worker(url, { type: 'module' });
      let id = 0;
      const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data, call = pending.get(response.id);
        if (!call) return;
        pending.delete(response.id);
        if (response.ok) call.resolve(response.result); else call.reject(new Error(`${response.error.code}: ${response.error.message}`));
      };
      const request = <K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> =>
        new Promise((resolve, reject) => { const number = ++id; pending.set(number, { resolve: value => resolve(value as Commands[K]['response']), reject }); worker.postMessage({ id: number, command, payload }); });
      try {
        let loaded;
        try { loaded = await request('LOAD', { manifestUrl: new URL('models/tinystories/manifest.json', location.href).href, backend }); }
        catch (error) { if (backend === 'webgl' && String(error).includes('UNSUPPORTED_BACKEND')) { records.push({ backend, unsupported: String(error) }); continue; } throw error; }
        if (loaded.manifest.modelRevision !== revision) throw new Error('Wrong TinyStories checkpoint loaded');
        const resources = await request('RESOURCES', {});
        let maximum = 0;
        for (const example of cases) {
          const run = await request('BASELINE', { tokenIds: example.tokenIds });
          const position = example.tokenIds.length - 1;
          const compare = (actual: Float32Array, expected: number[]) => expected.forEach((value, i) => {
            const error = Math.abs(actual[i] - value); maximum = Math.max(maximum, error);
            if (!Number.isFinite(actual[i]) || error > 2e-5 + 1e-5 * Math.abs(value)) throw new Error(`Text logit ${i} differs: ${error}`);
          });
          compare((await request('INSPECT', { runId: run.runId, reference: { stage: 'logits' }, selection: { position } })).values, example.logits);
          for (const variant of example.variants) {
            for (const executionPreference of ['auto', 'full'] as const) {
              const changed = await request('COMPARE', { baselineRunId: run.runId, interventions: [variant.intervention as Commands['COMPARE']['request']['interventions'][0]], predictionPosition: position, targetTokenId: 259, executionPreference });
              compare((await request('INSPECT', { runId: changed.modifiedRunId, reference: { stage: 'logits' }, selection: { position } })).values, variant.logits);
              await request('RELEASE', { runId: changed.modifiedRunId });
            }
          }
          await request('RELEASE', { runId: run.runId });
        }
        const probe = await request('PROBE', {});
        const final = await request('RESOURCES', {});
        records.push({ backend: loaded.backend, maximum, probe, retainedTensors: final.tensorCount - resources.tensorCount });
      } finally { worker.terminate(); }
    }
    return records;
  }, { url: workerUrl, cases: fixture.cases, revision: fixture.modelRevision });
  for (const record of records) {
    if ('unsupported' in record) testInfo.annotations.push({ type: 'unsupported-webgl', description: record.unsupported! });
    else { expect(record.retainedTensors).toBe(0); expect(record.probe?.retainedTensorDelta).toBe(0); }
  }
  await testInfo.attach('tinystories-numerical-report', { body: JSON.stringify({ userAgent: await page.evaluate(() => navigator.userAgent), revision: fixture.modelRevision, records }, null, 2), contentType: 'application/json' });
});
