import { expect, test } from '@playwright/test';
import type { Command, Commands, WorkerResponse } from '../../src/workers/protocol';
import type { ActivationReference, HeadScaleIntervention } from '../../src/engine/contracts';
import type { ProbeFixture } from '../../src/engine/probe';

test('production CPU/WebGL suffix equivalence, eviction, reload, and 1,000-edit resource stress', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  let workerUrl = '';
  page.on('request', request => { if (/model\.worker-.*\.js/u.test(request.url())) workerUrl = request.url(); });
  await page.goto('');
  await expect(page.getByTestId('execution-strategy')).toHaveText('Cached suffix · layer 1');
  const report = await page.evaluate(async url => {
    const worker = new Worker(url, { type: 'module' });
    let nextId = 0;
    const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data, call = pending.get(response.id);
      if (!call) return;
      pending.delete(response.id);
      if (response.ok) call.resolve(response.result);
      else call.reject(new Error(`${response.error.code}: ${response.error.message}`));
    };
    const request = <K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> =>
      new Promise((resolve, reject) => {
        const id = ++nextId;
        pending.set(id, { resolve: value => resolve(value as Commands[K]['response']), reject });
        worker.postMessage({ id, command, payload });
      });
    const manifestUrl = new URL('models/verification/manifest.json', window.location.href).href;
    const fixture = await (await fetch(new URL('probe.json', manifestUrl))).json() as ProbeFixture;
    const tokens = [1, 5, 9, 3, 7];
    const patch: HeadScaleIntervention = { id: 'scale', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'all' } };
    const reports = [];
    function close(a: ArrayLike<number>, b: ArrayLike<number>, label: string): number {
      if (a.length !== b.length) throw new Error(`Shape mismatch: ${label}`);
      let maximum = 0;
      for (let i = 0; i < a.length; i++) {
        const error = Math.abs(a[i] - b[i]);
        if (!Number.isFinite(a[i]) || error > 2e-5 + 1e-5 * Math.abs(b[i])) throw new Error(`${label}[${i}]: error ${error}`);
        maximum = Math.max(maximum, error);
      }
      return maximum;
    }
    async function inspect(runId: string, reference: ActivationReference) {
      return (await request('INSPECT', { runId, reference })).values;
    }
    try {
      for (const backend of ['cpu', 'webgl'] as const) {
        try { await request('LOAD', { backend, manifestUrl }); }
        catch (error) {
          if (backend !== 'webgl' || !(error instanceof Error) || !error.message.startsWith('UNSUPPORTED_BACKEND')) throw error;
          reports.push({ backend, status: 'unsupported', reason: error.message }); continue;
        }
        const baseline = await request('BASELINE', { tokenIds: tokens });
        let logitError = 0, residualError = 0, independentError = 0, cachedCases = 0;
        const selections: HeadScaleIntervention[][] = [];
        for (const layer of [0, 1]) {
          for (const factor of [0, .5, 2]) for (const scope of [{ kind: 'all' as const }, { kind: 'indices' as const, indices: [0, 4] }])
            selections.push([{ ...patch, layer, factor, tokens: scope }]);
          selections.push([{ ...patch, layer }, { ...patch, id: 'other', layer, head: 1, factor: 2, tokens: { kind: 'indices', indices: [0, 4] } }]);
        }
        for (const interventions of selections) {
          const input = { baselineRunId: baseline.runId, interventions, topK: 16 };
          const cached = await request('COMPARE', input);
          const full = await request('COMPARE', { ...input, executionPreference: 'full' });
          if (cached.execution.strategy !== 'cached-suffix' || full.execution.strategy !== 'full') throw new Error('Incorrect execution plan.');
          cachedCases++;
          const a = await inspect(cached.modifiedRunId, { stage: 'logits' });
          const b = await inspect(full.modifiedRunId, { stage: 'logits' });
          logitError = Math.max(logitError, close(a, b, 'cached/full logits'));
          const layer = interventions[0].layer;
          residualError = Math.max(residualError, close(await inspect(cached.modifiedRunId, { layer, stage: 'attention.residual' }),
            await inspect(full.modifiedRunId, { layer, stage: 'attention.residual' }), 'cached/full residual'));
          if (layer === 0 && interventions.length === 1 && interventions[0].factor === .5 && interventions[0].tokens.kind === 'all') {
            const expected = fixture.cases.find(example => example.name === 'length-5-case-2')!;
            independentError = close(a, expected.activations.logits.values, 'independent PyTorch logits');
          }
          await request('RELEASE', { runId: cached.modifiedRunId }); await request('RELEASE', { runId: full.modifiedRunId });
        }
        const cross = await request('COMPARE', { baselineRunId: baseline.runId, interventions: [patch, { ...patch, layer: 1, head: 1 }] });
        if (cross.execution.strategy !== 'full') throw new Error('Cross-layer edits reused a downstream baseline.');
        await request('RELEASE', { runId: cross.modifiedRunId });

        const beforeStress = await request('RESOURCES', {});
        for (let i = 0; i < 1000; i++) {
          const factor = [0, .5, 1, 2][i % 4];
          const edited = await request('COMPARE', { baselineRunId: baseline.runId, interventions: [{ ...patch, factor, layer: i % 2 }] });
          if (edited.execution.strategy !== (factor === 1 ? 'identity' : 'cached-suffix')) throw new Error('Stress execution plan changed unexpectedly.');
          await request('RELEASE', { runId: edited.modifiedRunId });
        }
        const afterStress = await request('RESOURCES', {});
        if (afterStress.tensorCount !== beforeStress.tensorCount || afterStress.cache.logicalBytes !== beforeStress.cache.logicalBytes || afterStress.cache.pinnedEntries)
          throw new Error('Resources grew across 1,000 edits.');
        await request('RELEASE', { runId: baseline.runId });
        const afterRelease = await request('RESOURCES', {});
        if (afterRelease.cache.logicalBytes !== 0 || afterRelease.tensorCount !== beforeStress.tensorCount - 4) throw new Error('Baseline release leaked tensors.');

        await request('LOAD', { backend, manifestUrl, cacheBudgetBytes: 640 });
        const old = await request('BASELINE', { tokenIds: tokens });
        const saved = await request('COMPARE', { baselineRunId: old.runId, interventions: [patch] });
        const beforeEviction = await inspect(saved.modifiedRunId, { stage: 'logits' });
        const newer = await request('BASELINE', { tokenIds: [2, 5, 9, 3, 7] });
        const stats = await request('RESOURCES', {});
        if (stats.cache.logicalBytes !== 640 || stats.cache.baselines !== 1) throw new Error('LRU cache exceeded its budget.');
        const fallback = await request('COMPARE', { baselineRunId: old.runId, interventions: [patch] });
        if (fallback.execution.strategy !== 'full') throw new Error('Evicted baseline did not fall back.');
        close(await inspect(saved.modifiedRunId, { stage: 'logits' }), beforeEviction, 'replay after eviction');
        await request('RELEASE', { runId: newer.runId });
        if ((await request('RESOURCES', {})).cache.logicalBytes !== 0) throw new Error('Evicted/released cache retained tensors.');
        await request('LOAD', { backend, manifestUrl, cacheBudgetBytes: 0 });
        let invalidated = false;
        try { await request('COMPARE', { baselineRunId: old.runId, interventions: [patch] }); }
        catch (error) { invalidated = error instanceof Error && error.message.startsWith('STALE_SESSION'); }
        if (!invalidated) throw new Error('Reload reused a stale run.');
        const uncached = await request('BASELINE', { tokenIds: tokens });
        if ((await request('COMPARE', { baselineRunId: uncached.runId, interventions: [patch] })).execution.strategy !== 'full') throw new Error('Zero-budget run used a cache.');
        reports.push({ backend, status: 'passed', cachedCases, logitError, residualError, independentError,
          beforeStress, afterStress, afterRelease, stressComparisons: 1000 });
      }
      return { userAgent: navigator.userAgent, reports };
    } finally { worker.terminate(); }
  }, workerUrl);
  expect(report.reports[0].status).toBe('passed');
  if (testInfo.project.name === 'chromium') expect(report.reports[1].status).toBe('passed');
  await testInfo.attach('cache-verification-results', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
});
