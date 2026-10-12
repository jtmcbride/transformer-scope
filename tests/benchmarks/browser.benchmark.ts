import { execFileSync } from 'node:child_process';
import { cpus, platform, release, totalmem } from 'node:os';
import { expect, test } from '@playwright/test';
import type { Command, Commands, WorkerResponse } from '../../src/workers/protocol';

const hardware = { machineModel: platform() === 'darwin' ? execFileSync('sysctl', ['-n', 'hw.model'], { encoding: 'utf8' }).trim() : null,
  cpuModel: cpus()[0]?.model ?? 'unknown', logicalCpuCount: cpus().length, memoryGiB: totalmem() / 1024 ** 3,
  os: platform() === 'darwin' ? `macOS ${execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim()}` : `${platform()} ${release()}` };

for (const model of ['verification', 'retrieval'] as const) for (const backend of ['cpu', 'webgl'] as const) {
  test(`${model} ${backend}: cold page, warm interaction, and full/cached RPC`, async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    let workerUrl = '';
    page.on('request', request => { if (/model\.worker-.*\.js/u.test(request.url())) workerUrl = request.url(); });
    await page.addInitScript(() => {
      const observation = new MutationObserver(() => {
        if (document.querySelector('.status')?.textContent?.startsWith('Showing')) {
          (window as unknown as { benchmarkReadyMs: number }).benchmarkReadyMs = performance.now(); observation.disconnect();
        }
      });
      observation.observe(document, { childList: true, subtree: true, characterData: true });
    });
    await page.goto(`?model=${model}&backend=${backend}`);
    await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled({ timeout: 60_000 });
    await expect(page.getByLabel('Backend')).toHaveValue(backend);
    const coldReadyMs = await page.evaluate(() => (window as unknown as { benchmarkReadyMs: number }).benchmarkReadyMs);
    const ui = [];
    await page.getByLabel('Head', { exact: true }).selectOption('0');
    for (const layer of [0, 1]) {
      await page.getByLabel('Layer', { exact: true }).selectOption(String(layer));
      await expect(page.getByRole('button', { name: 'Export experiment' })).toBeEnabled();
      const samples = await page.evaluate(async () => {
        const input = document.getElementById('strength') as HTMLInputElement;
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
        const samples: { domCommitMs: number; frameOpportunityMs: number }[] = [];
        const offset = Number(input.value) === .25 ? 1 : 0;
        for (let i = 0; i < 45; i++) {
          const factor = [.25, .5, .75, 1.5][(i + offset) % 4];
          const started = performance.now();
          const sample = await new Promise<{ domCommitMs: number; frameOpportunityMs: number }>((resolve, reject) => {
            const expected = `Showing ${factor.toFixed(2)}×`;
            const timer = setTimeout(() => { observer.disconnect(); reject(new Error('Interaction did not complete.')); }, 30_000);
            const observer = new MutationObserver(() => {
              if (document.querySelector('.status')?.textContent !== expected) return;
              observer.disconnect(); clearTimeout(timer);
              const domCommitMs = performance.now() - started;
              requestAnimationFrame(() => requestAnimationFrame(() => resolve({ domCommitMs, frameOpportunityMs: performance.now() - started })));
            });
            observer.observe(document, { childList: true, subtree: true, characterData: true });
            setter.call(input, String(factor)); input.dispatchEvent(new Event('input', { bubbles: true }));
          });
          if (i >= 5) samples.push(sample);
        }
        return samples;
      });
      ui.push({ layer, samples });
    }
    const rpc = await page.evaluate(async ({ workerUrl, model, backend }) => {
      const worker = new Worker(workerUrl, { type: 'module' });
      let id = 0;
      const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data, call = pending.get(response.id);
        if (!call) return; pending.delete(response.id);
        if (response.ok) call.resolve(response.result); else call.reject(new Error(response.error.message));
      };
      const request = <K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> => new Promise((resolve, reject) => {
        const current = ++id; pending.set(current, { resolve: value => resolve(value as Commands[K]['response']), reject });
        worker.postMessage({ id: current, command, payload });
      });
      try {
        const loaded = await request('LOAD', { manifestUrl: new URL(`models/${model}/manifest.json`, window.location.href).href, backend });
        const symbols = (document.getElementById('sequence') as HTMLTextAreaElement).value.trim().split(/\s+/u);
        const tokens = [loaded.tokenizer.bosTokenId, ...symbols.map(symbol => loaded.tokenizer.tokens.indexOf(symbol))];
        const start = performance.now();
        const baseline = await request('BASELINE', { tokenIds: tokens });
        const baselineFirstMs = performance.now() - start;
        const layers = [];
        for (const layer of [0, 1]) {
          const samples = { full: [] as { rpcMs: number; workerMs: number }[], cached: [] as { rpcMs: number; workerMs: number }[] };
          for (let i = 0; i < 45; i++) for (const mode of (i % 2 ? ['cached', 'full'] : ['full', 'cached']) as ('full' | 'cached')[]) {
            const started = performance.now();
            const result = await request('COMPARE', { baselineRunId: baseline.runId, predictionPosition: tokens.length - 1,
              interventions: [{ id: 'benchmark', kind: 'head.scale', layer, head: 0, factor: .5, tokens: { kind: 'all' } }],
              executionPreference: mode === 'full' ? 'full' : 'auto' });
            const rpcMs = performance.now() - started;
            if (result.execution.strategy !== (mode === 'full' ? 'full' : 'cached-suffix')) throw new Error('Unexpected benchmark strategy.');
            if (i >= 5) samples[mode].push({ rpcMs, workerMs: result.execution.elapsedMs });
            await request('RELEASE', { runId: result.modifiedRunId });
          }
          layers.push({ layer, samples });
        }
        await request('RELEASE', { runId: baseline.runId });
        return { modelRevision: loaded.manifest.modelRevision, config: loaded.manifest.config, precision: loaded.precision, baselineFirstMs, sequenceLength: tokens.length, layers };
      } finally { worker.terminate(); }
    }, { workerUrl, model, backend });
    function summarize(values: number[]) {
      const sorted = [...values].sort((a, b) => a - b);
      return { count: sorted.length, medianMs: (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2,
        p95Ms: sorted[Math.ceil(sorted.length * .95) - 1] };
    }
    const result = { model, backend, browser: testInfo.project.name, userAgent: await page.evaluate(() => navigator.userAgent),
      hardware, coldReadyMs,
      ui: ui.map(item => ({ ...item, domCommit: summarize(item.samples.map(sample => sample.domCommitMs)), frameOpportunity: summarize(item.samples.map(sample => sample.frameOpportunityMs)) })),
      rpc: { ...rpc, layers: rpc.layers.map(item => ({ ...item, full: summarize(item.samples.full.map(sample => sample.rpcMs)), cached: summarize(item.samples.cached.map(sample => sample.rpcMs)) })) } };
    await testInfo.attach('browser-benchmark', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  });
}
