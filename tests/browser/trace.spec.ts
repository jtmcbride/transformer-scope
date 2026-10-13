import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import fixture from '../fixtures/trace-reference.json' with { type: 'json' };
import redirectFixture from '../fixtures/attention-trace-reference.json' with { type: 'json' };
import copyFixture from '../fixtures/residual-trace-reference.json' with { type: 'json' };
import type { Commands, Command, WorkerResponse } from '../../src/workers/protocol';
import type { EffectTraceReport } from '../../src/engine/trace';

async function recordCommands(page: Page) {
  await page.addInitScript(() => {
    const Original = window.Worker, commands: string[] = [];
    Object.assign(window, { __traceCommands: commands });
    window.Worker = class extends Original {
      postMessage(message: unknown, options: Transferable[] | StructuredSerializeOptions = []) {
        commands.push((message as { command: string }).command);
        if (Array.isArray(options)) super.postMessage(message, options); else super.postMessage(message, options);
      }
    };
  });
}
const calls = (page: Page, command: string) => page.evaluate(cmd => (window as unknown as { __traceCommands: string[] }).__traceCommands.filter(item => item === cmd).length, command);
async function ready(page: Page) {
  await expect(page.getByTestId('trace-step')).toContainText('Step 1 of', { timeout: 30_000 });
}

test('the guided large-effect trace distinguishes attention, output and diagnostic changes without recomputing during navigation', async ({ page }, testInfo) => {
  await recordCommands(page);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  expect(await calls(page, 'TRACE')).toBe(0);
  const compareCount = await calls(page, 'COMPARE');
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await ready(page);
  await expect(page.getByTestId('trace-first-flip')).toContainText('Layer 2 after attention');
  const edited = page.getByRole('button', { name: 'Inspect Layer 1 Head 4', exact: true });
  await expect(edited).toHaveAttribute('data-trace-route', 'prediction');
  await expect(edited).toHaveAttribute('data-trace-change', 'changed');
  await expect(edited).toContainText('Attention: Exact match');
  await expect(edited).toContainText('Output: Changed');
  for (let index = 1; index <= 4; index++) {
    await page.getByRole('button', { name: 'Next', exact: true }).click();
    await expect(page.getByTestId('trace-step')).toContainText(`Step ${index + 1} of`);
  }
  await expect(page.getByTestId('node-readout')).toContainText('Preferred token: 3 → 3');
  await expect(page.getByTestId('node-readout')).toContainText('Diagnostic readout');
  await page.getByRole('button', { name: 'Inspect Layer 2 Residual after attention', exact: true }).click();
  await expect(page.getByTestId('node-readout')).toContainText('Preferred token: 3 → 0');
  await page.getByLabel('Trace inspection position').selectOption('0');
  await expect(page.getByTestId('trace-step')).toContainText('position 0');
  await expect(page.getByTestId('trace-handoff')).toContainText('One route among all edited positions');
  await page.getByLabel('Trace inspection position').selectOption('15');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('node-readout')).toContainText('Model prediction');
  await expect(page.getByTestId('baseline-answer')).toHaveText('3');
  await expect(page.getByTestId('modified-answer')).toHaveText('0');
  expect(await calls(page, 'TRACE')).toBe(1);
  expect(await calls(page, 'COMPARE')).toBe(compareCount);
  await page.screenshot({ path: testInfo.outputPath('trace-desktop.png') });
  await page.getByRole('button', { name: 'Exit trace', exact: true }).click();
  await expect(page.locator('.flow-node[data-trace-change]')).toHaveCount(0);
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
});

test('position-aware tracing explains a small probability effect, a local no-path edit, and an identity', async ({ page }) => {
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByLabel('Prepared experiment').selectOption('small');
  await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled();
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await ready(page);
  await expect(page.getByTestId('trace-handoff')).toContainText('Position 0 → 15');
  await expect(page.getByTestId('trace-path')).toContainText('through Layer 2 attention');
  await expect(page.getByLabel('Trace inspection position')).toHaveValue('0');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('readout-delta')).not.toHaveText('Δ log p: 0 nats');
  await expect(page.getByTestId('readout-delta')).toContainText('e-');
  await page.getByLabel('Prepared experiment').selectOption('unreachable');
  await ready(page);
  await expect(page.getByTestId('trace-path')).toContainText('no computational path');
  await expect(page.getByTestId('trace-handoff')).toHaveCount(0);
  const local = page.getByRole('button', { name: 'Inspect Layer 2 Head 1', exact: true });
  await expect(local).toHaveAttribute('data-trace-route', 'local');
  await expect(local).toHaveAttribute('data-trace-change', 'changed');
  await page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true }).click();
  await expect(page.getByTestId('readout-delta')).toHaveText('Δ log p: 0 nats');
  await expect(page.getByRole('button', { name: 'Inspect Readout → prediction', exact: true })).toHaveAttribute('data-trace-change', 'exact');
  await page.getByRole('button', { name: 'Reset to baseline' }).click();
  await ready(page);
  await expect(page.getByTestId('trace-path')).toContainText('no effective edit');
  await expect(page.locator('.trace-scope')).toContainText('Selected position 0');
  await expect(page.locator('.flow-node[data-trace-route=prediction], .flow-node[data-trace-route=local]')).toHaveCount(0);
  await expect(page.locator('.flow-node[data-trace-change=changed], .flow-node[data-trace-change=small]')).toHaveCount(0);
});

test('late traces cannot overwrite rapid edits, checkpoint switching or a fresh worker', async ({ page }) => {
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
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await expect.poll(() => calls(page, 'TRACE')).toBe(1);
  const slider = page.getByRole('slider', { name: 'Head strength', exact: true });
  await slider.fill('0.25'); await slider.fill('0.37'); await slider.fill('0.73');
  await expect(page.locator('.status')).toHaveText('Showing 0.73×');
  await ready(page);
  await expect(page.locator('.trace-scope')).toContainText('0.73×');
  expect(await calls(page, 'TRACE')).toBeLessThanOrEqual(2);
  await page.getByLabel('Model', { exact: true }).selectOption('verification');
  await expect(page.locator('.status')).toHaveText('Showing 0.50×');
  await ready(page);
  await expect(page.locator('.trace-scope')).toContainText('Head 1 · 0.5×');
  await page.getByRole('button', { name: 'Restart computation worker on CPU' }).click();
  await ready(page);
  await expect(page.locator('.trace-scope')).toContainText('Head 1 · 0.5×');
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('CPU and supported WebGL traces match all independent fixtures and clean up their resources', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  let workerUrl = '';
  page.on('request', request => { if (/model\.worker-.*\.js/u.test(request.url())) workerUrl = request.url(); });
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  const results = await page.evaluate(async ({ url, cases }) => {
    const worker = new Worker(url, { type: 'module' }), pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    let id = 0;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data, call = pending.get(message.id);
      if (!call) return; pending.delete(message.id);
      if (message.ok) call.resolve(message.result); else call.reject(new Error(`${message.error.code}: ${message.error.message}`));
    };
    const request = <K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> => new Promise((resolve, reject) => {
      const current = ++id; pending.set(current, { resolve: value => resolve(value as Commands[K]['response']), reject }); worker.postMessage({ id: current, command, payload });
    });
    const reports: { backend: string; checkpoint: string; status: string; detail?: string; count?: number }[] = [];
    const close = (a: number[], b: number[], atol = 2e-5, rtol = 1e-5) => {
      if (a.length !== b.length || a.some((value, i) => !Number.isFinite(value) || Math.abs(value - b[i]) > atol + rtol * Math.abs(b[i]))) throw new Error('Trace reference mismatch.');
    };
    try {
      for (const backend of ['cpu', 'webgl'] as const) for (const checkpoint of ['retrieval', 'verification']) {
        let revision = '';
        try { revision = (await request('LOAD', { manifestUrl: new URL(`models/${checkpoint}/manifest.json`, window.location.href).href, backend })).manifest.modelRevision; }
        catch (error) {
          if (backend === 'webgl' && error instanceof Error && error.message.startsWith('UNSUPPORTED_BACKEND')) { reports.push({ backend, checkpoint, status: 'unsupported', detail: error.message }); continue; }
          throw error;
        }
        const selected = cases.filter(item => item.modelRevision === revision);
        for (const item of selected) {
          const baseline = await request('BASELINE', { tokenIds: item.tokens });
          let sourceId: string | undefined;
          let patch = item.intervention as Commands['COMPARE']['request']['interventions'][number];
          if ('sourceTokens' in item && patch.kind === 'residual.replace') {
            sourceId = (await request('BASELINE', { tokenIds: item.sourceTokens })).runId;
            const read = await request('INSPECT', { runId: sourceId, reference: { layer: patch.layer, stage: patch.stage }, selection: { position: patch.position } });
            patch = { ...patch, values: Array.from(read.values) };
          }
          const comparison = await request('COMPARE', { baselineRunId: baseline.runId, interventions: [patch], predictionPosition: item.predictionPosition });
          const before = await request('RESOURCES', {});
          const trace: EffectTraceReport = await request('TRACE', { baselineRunId: baseline.runId, modifiedRunId: comparison.modifiedRunId, predictionPosition: item.predictionPosition });
          const after = await request('RESOURCES', {});
          if (after.tensorCount !== before.tensorCount || after.logicalTensorBytes !== before.logicalTensorBytes || after.cache.logicalBytes !== before.cache.logicalBytes) throw new Error('Trace retained tensors.');
          if (trace.hasComputationalPath !== comparison.hasComputationalPath || trace.nodes.length !== item.nodes.length) throw new Error('Trace route/shape mismatch.');
          trace.nodes.forEach((node, i) => close([node.baselineNorm, node.modifiedNorm, node.maxDifference, node.differenceNorm],
            [item.nodes[i].baselineNorm, item.nodes[i].modifiedNorm, item.nodes[i].maxDifference, item.nodes[i].differenceNorm], .0005, 2e-5));
          trace.readouts.slice(0, -1).forEach((row, i) => { close(row.baselineLogits, item.baselineReadouts[i]); close(row.modifiedLogits, item.modifiedReadouts[i]); });
          close(trace.readouts.at(-1)!.modifiedLogits, item.modifiedLogits);
          if ('attentionRow' in item) {
            close(trace.attentionRow!.baselineWeights, item.attentionRow.baselineWeights);
            close(trace.attentionRow!.modifiedWeights, item.attentionRow.modifiedWeights);
            if (trace.attentionRow!.scores.change !== 'exact') throw new Error('A redirect changed its own scores.');
          }
          if ('residualRow' in item && patch.kind === 'residual.replace') {
            close(trace.residualRow!.baselineValues, item.residualRow.baselineValues);
            close(trace.residualRow!.copiedValues, item.residualRow.copiedValues);
            if (trace.residualRow!.copiedValues.some((value, i) => value !== Math.fround(patch.values[i]))) throw new Error('The trace did not capture the copied row.');
          }
          if (sourceId) await request('RELEASE', { runId: sourceId });
          await request('RELEASE', { runId: comparison.modifiedRunId }); await request('RELEASE', { runId: baseline.runId });
        }
        reports.push({ backend, checkpoint, status: 'passed', count: selected.length });
      }
      return reports;
    } finally { worker.terminate(); }
  }, { url: workerUrl, cases: [...fixture.cases, ...redirectFixture.cases, ...copyFixture.cases] });
  for (const result of results) {
    if (result.status === 'unsupported') testInfo.annotations.push({ type: 'unsupported-webgl', description: result.detail! });
    if (result.backend === 'cpu' || testInfo.project.name === 'chromium') expect(result.status).toBe('passed');
  }
  await testInfo.attach('trace-backend-results', { body: JSON.stringify(results, null, 2), contentType: 'application/json' });
});

test('mobile and keyboard trace navigation keep controls and prediction visible', async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const width of [360, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('');
    await expect(page.locator('.status')).toHaveText('Showing 0.00×');
    await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
    await ready(page);
    await page.getByRole('button', { name: 'Next', exact: true }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
    for (const selector of ['.trace-navigation', '.prediction-summary']) {
      const contained = await page.locator(selector).evaluate(element => {
        const box = element.getBoundingClientRect(); return box.left >= -1 && box.right <= innerWidth + 1 && box.top >= 0 && box.bottom <= innerHeight;
      });
      expect(contained).toBe(true);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    if (width === 390) await page.screenshot({ path: testInfo.outputPath('trace-mobile.png') });
    await page.getByRole('button', { name: 'Exit trace', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Follow the change', exact: true })).toBeEnabled();
  }
});

test('input replacement and replay reset the trace, while export and copy drafts preserve the completed experiment', async ({ page }) => {
  await recordCommands(page);
  await page.goto('');
  await expect(page.locator('.status')).toHaveText('Showing 0.00×');
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await ready(page);
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment' }).click();
  const saved = await readFile((await (await download).path())!, 'utf8');
  await expect(page.getByTestId('trace-step')).toContainText('Step 2 of');
  expect(await calls(page, 'TRACE')).toBe(1);
  await page.getByLabel('Binding 1 value').selectOption('2');
  await page.getByLabel('Ask for', { exact: true }).selectOption('A');
  await page.getByRole('button', { name: 'Run input' }).click();
  await ready(page);
  await expect(page.getByTestId('baseline-answer')).toHaveText('2');
  await expect(page.locator('.trace-scope')).toContainText('Prediction position 15');
  await page.getByLabel('Import experiment').setInputFiles({ name: 'trace-replay.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(page.getByTestId('replay-report')).toContainText('Replay matches');
  await ready(page);
  await expect(page.getByTestId('baseline-answer')).toHaveText('3');
  await expect(page.getByTestId('modified-answer')).toHaveText('0');
  await page.getByLabel('Intervention type').selectOption('attention');
  await expect(page.locator('.status')).toHaveText('Showing redirected attention');
  await ready(page);
  await expect(page.getByTestId('trace-step')).toContainText('attention weights');
  await page.getByLabel('Intervention type').selectOption('patch');
  await expect(page.locator('.status')).toHaveText('Showing redirected attention');
  await page.getByRole('button', { name: 'Copy from source', exact: true }).click();
  await expect(page.locator('.status')).toHaveText('Showing patched residual');
  await ready(page);
  await expect(page.getByTestId('trace-step')).toContainText('Layer input');
  await expect(page.getByTestId('graph-copy-connection')).toHaveCount(1);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByRole('button', { name: 'Exit trace', exact: true }).click();
});

test('a lost WebGL context during a trace discards measurements and restarts tracing on CPU', async ({ page }, testInfo) => {
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `{
      const contexts = [];
      if (typeof OffscreenCanvas !== 'undefined') {
        const original = OffscreenCanvas.prototype.getContext;
        OffscreenCanvas.prototype.getContext = function(kind, ...args) {
          const context = original.call(this, kind, ...args);
          if (context && String(kind).includes('webgl')) contexts.push(context);
          return context;
        };
      }
      self.addEventListener('message', event => {
        if (event.data?.command === 'TRACE') setTimeout(() => contexts.forEach(context => context.getExtension('WEBGL_lose_context')?.loseContext()), 0);
      });
    }\n${await response.text()}` });
  });
  await page.goto('?backend=webgl');
  await expect(page.locator('.status').filter({ hasText: 'Showing 0.00×' }).or(page.getByRole('alert'))).toBeVisible();
  const unsupported = page.getByRole('alert').filter({ hasText: 'WebGL worker unavailable' });
  if (await unsupported.count()) { testInfo.annotations.push({ type: 'unsupported-webgl', description: await unsupported.innerText() }); return; }
  await page.getByRole('button', { name: 'Follow the change', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('WebGL context lost');
  await expect(page.getByTestId('trace-step')).toHaveText('Worker session ended');
  await expect(page.getByTestId('example-guide')).toContainText('Last completed example · session ended');
  await expect(page.getByRole('button', { name: 'Trace this example', exact: true })).toBeDisabled();
  await expect(page.locator('.flow-node[data-trace-change]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Restart on CPU', exact: true }).click();
  await ready(page);
  await expect(page.getByLabel('Backend')).toHaveValue('cpu');
  await expect(page.getByTestId('trace-first-flip')).toContainText('Layer 2 after attention');
  await expect(page.getByRole('alert')).toHaveCount(0);
});
