import { EngineError, tokenize, type HeadScaleIntervention } from '../engine/contracts';
import { checkReplay, createExperiment, parseExperiment, validateExperiment } from '../engine/experiment';
import { fetchExperiments, validateExperiments } from '../experiments';
import { button, change, openFrame, select, settled, statusText, waitFor, type Frame } from '../embedded/frame';
import { ModelClient } from '../workers/client';
import type { Commands } from '../workers/protocol';
import {
  benchmarkHead, compareLogits, recipeExpectations, rpcFactor, runCheck, summarize, summarizeChecks, warmFactors,
  type CheckRecord, type CombinationRecord, type EvidenceClass, type HarnessBackend, type HarnessModel, type HarnessSettings, type ValidationReport,
} from './report';

type Loaded = Commands['LOAD']['response'];
type Progress = (message: string) => void;
interface ReleaseStamp { commit: string; builtAt: string; sourceTreeDirty: boolean; basePath: string; models: Record<string, { modelId: string; modelRevision: string }> }
/** The app's default verification input; retrieval uses the first shipped recipe, as the app and benchmark do. */
const verificationInput = 'A E I C G';
const message = (error: unknown) => error instanceof Error ? error.message : String(error);

async function readRelease(base: URL): Promise<{ record: CheckRecord; stamp: ReleaseStamp | null }> {
  let stamp: ReleaseStamp | null = null;
  const record = await runCheck('release-identity', async () => {
    const response = await fetch(new URL('release.json', base), { cache: 'no-store' });
    if (!response.ok) throw new Error(`release.json returned HTTP ${response.status}; this page is not a stamped release build.`);
    stamp = await response.json() as ReleaseStamp;
    if (!/^[0-9a-f]{40}$/u.test(stamp.commit)) throw new Error('release.json does not contain a full source commit.');
    return { detail: `${stamp.commit.slice(0, 7)}${stamp.sourceTreeDirty ? ' (built from a modified working tree)' : ''}`,
      data: { commit: stamp.commit, builtAt: stamp.builtAt, sourceTreeDirty: stamp.sourceTreeDirty, basePath: stamp.basePath, models: stamp.models } };
  });
  return { record, stamp };
}

function timerResolution(): number | null {
  // Smallest positive step observed in a short sampling loop; coarse timers affect every millisecond reading.
  let smallest = Infinity, previous = performance.now();
  const deadline = previous + 25;
  for (let i = 0; i < 200_000; i++) {
    const now = performance.now();
    if (now > previous) { smallest = Math.min(smallest, now - previous); previous = now; }
    if (now > deadline) break;
  }
  return Number.isFinite(smallest) ? smallest : null;
}

function mainThreadWebgl(): Record<string, unknown> {
  try {
    const canvas = document.createElement('canvas');
    const context = (canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as WebGLRenderingContext | null;
    if (!context) return { available: false };
    const debug = context.getExtension('WEBGL_debug_renderer_info');
    const result = { available: true, version: context.getParameter(context.VERSION),
      vendor: debug ? context.getParameter(debug.UNMASKED_VENDOR_WEBGL) : context.getParameter(context.VENDOR),
      renderer: debug ? context.getParameter(debug.UNMASKED_RENDERER_WEBGL) : context.getParameter(context.RENDERER),
      unmasked: !!debug, scope: 'main-thread context; the computation worker creates its own context' };
    context.getExtension('WEBGL_lose_context')?.loseContext();
    return result;
  } catch (error) { return { available: false, error: message(error) }; }
}

export async function captureEnvironment(): Promise<Record<string, unknown>> {
  const navigatorRecord = navigator as Navigator & { deviceMemory?: number; userAgentData?: { brands: unknown; mobile: boolean; platform: string;
    getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>> } };
  let userAgentData: unknown = null;
  if (navigatorRecord.userAgentData) {
    const { brands, mobile, platform } = navigatorRecord.userAgentData;
    userAgentData = { brands, mobile, platform };
    try {
      userAgentData = { ...(userAgentData as object), ...await navigatorRecord.userAgentData.getHighEntropyValues?.(['platformVersion', 'model', 'architecture', 'fullVersionList']) };
    } catch (error) { userAgentData = { ...(userAgentData as object), highEntropyError: message(error) }; }
  }
  return {
    userAgent: navigator.userAgent, userAgentData, hardwareConcurrency: navigator.hardwareConcurrency ?? null, deviceMemoryGiB: navigatorRecord.deviceMemory ?? null,
    maxTouchPoints: navigator.maxTouchPoints, coarsePointer: matchMedia('(pointer: coarse)').matches, hoverNone: matchMedia('(hover: none)').matches,
    screen: { width: screen.width, height: screen.height, orientation: screen.orientation?.type ?? null }, devicePixelRatio,
    viewport: { width: innerWidth, height: innerHeight }, offscreenCanvas: typeof OffscreenCanvas !== 'undefined',
    timerResolutionMs: timerResolution(), mainThreadWebgl: mainThreadWebgl(),
    note: 'Browser reports only. Device model, power state, and thermal state must be entered by the observer.',
  };
}

async function interfaceChecks(host: HTMLElement, model: HarnessModel, backend: HarnessBackend, settings: HarnessSettings, base: URL, progress: Progress): Promise<CheckRecord[]> {
  const names = ['ui-cold-readiness', 'ui-warm-interaction', 'ui-baseline-reset', 'ui-cpu-recovery'];
  let target: Frame | null = null;
  const checks: CheckRecord[] = [];
  const skipRemaining = (detail: string) => names.slice(checks.length).forEach(name => checks.push({ name, status: 'skipped', detail }));
  try {
    progress(`${model}/${backend}: loading the application page`);
    checks.push(await runCheck('ui-cold-readiness', async () => {
      target = await openFrame(host, new URL(`./?model=${model}&backend=${backend}`, base), `Transformer Microscope under measurement (${model}, ${backend})`);
      const frame = target;
      const ready = await waitFor(frame, () => statusText(frame.doc).startsWith('Showing') || !!frame.doc.querySelector('[role="alert"]'), 'the first result or an error');
      const alert = frame.doc.querySelector('[role="alert"]')?.textContent ?? '';
      if (!statusText(frame.doc).startsWith('Showing')) {
        if (backend === 'webgl' && alert.startsWith('WebGL worker unavailable')) return { status: 'unsupported', detail: alert };
        throw new Error(alert || 'The application did not show a result.');
      }
      await waitFor(frame, () => settled(frame.doc), 'default inspectors');
      // WebKit throttles animation frames for offscreen iframes even when the parent is visible.
      frame.frame.scrollIntoView({ block: 'center', behavior: 'instant' });
      const actual = select(frame.doc, 'Backend').value;
      if (actual !== backend) throw new Error(`Requested ${backend}; the application reports ${actual}.`);
      const weights = frame.win.performance.getEntriesByType('resource').filter(entry => entry.name.endsWith(`models/${model}/weights.bin`))
        .map(entry => { const timing = entry as PerformanceResourceTiming; return { transferSize: timing.transferSize ?? null, encodedBodySize: timing.encodedBodySize ?? null, durationMs: timing.duration }; });
      return { detail: `${ready.at.toFixed(1)} ms${ready.alreadyTrue ? ' (upper bound: complete before observation began)' : ''}`,
        data: { workspaceView: 'visual', coldReadyMs: ready.at, observationStartedAfterCompletion: ready.alreadyTrue, actualBackend: actual, weightsFetch: weights,
          definition: 'Embedded page navigation start until the first completed comparison is committed to the DOM. HTTP, filesystem, driver, and shader caches are not reset; transferSize 0 usually means a cache hit.' } };
    }));
    if (checks[0].status !== 'passed') { skipRemaining(checks[0].status === 'unsupported' ? 'Backend unsupported.' : 'The application did not become ready.'); return checks; }
    const frame = target!;
    const layers = Number(select(frame.doc, 'Layer').options.length);

    checks.push(await runCheck('ui-warm-interaction', async () => {
      await change(frame, select(frame.doc, 'Head'), String(benchmarkHead));
      await change(frame, select(frame.doc, 'Apply at'), 'all');
      const results = [];
      for (const layer of [0, layers - 1]) {
        progress(`${model}/${backend}: timing ${settings.measuredSamples} interface edits at layer ${layer + 1}`);
        await change(frame, select(frame.doc, 'Layer'), String(layer));
        const input = frame.doc.getElementById('strength') as HTMLInputElement;
        const setter = Object.getOwnPropertyDescriptor(frame.win.HTMLInputElement.prototype, 'value')!.set!;
        const offset = Number(input.value) === warmFactors[0] ? 1 : 0;
        const samples: { domCommitMs: number; frameOpportunityMs: number }[] = [];
        for (let i = 0; i < settings.warmupSamples + settings.measuredSamples; i++) {
          const factor = warmFactors[(i + offset) % warmFactors.length];
          frame.frame.scrollIntoView({ block: 'center', behavior: 'instant' });
          const expected = `Showing ${factor.toFixed(2)}×`;
          const started = frame.win.performance.now();
          const sample = await new Promise<{ domCommitMs: number; frameOpportunityMs: number }>((resolve, reject) => {
            const timer = setTimeout(() => { observer.disconnect(); reject(new Error(`Edit to ${factor} did not complete within 30 s.`)); }, 30_000);
            const observer = new frame.win.MutationObserver(() => {
              if (statusText(frame.doc) !== expected) return;
              observer.disconnect();
              const domCommitMs = frame.win.performance.now() - started;
              frame.win.requestAnimationFrame(() => frame.win.requestAnimationFrame(() => {
                clearTimeout(timer); resolve({ domCommitMs, frameOpportunityMs: frame.win.performance.now() - started });
              }));
            });
            observer.observe(frame.doc, { childList: true, subtree: true, characterData: true });
            setter.call(input, String(factor)); input.dispatchEvent(new frame.win.Event('input', { bubbles: true }));
          });
          if (i >= settings.warmupSamples) samples.push(sample);
        }
        results.push({ layer, domCommit: summarize(samples.map(sample => sample.domCommitMs)), frameOpportunity: summarize(samples.map(sample => sample.frameOpportunityMs)), samples });
      }
      return { detail: results.map(result => `layer ${result.layer + 1}: commit p95 ${result.domCommit.p95Ms.toFixed(1)} ms, frame opportunity p95 ${result.frameOpportunity.p95Ms.toFixed(1)} ms`).join('; '),
        data: { workspaceView: 'visual', head: benchmarkHead, scope: 'all positions', factors: warmFactors, layers: results,
          definition: 'Synthetic in-page input event to the completed result DOM commit (scheduling, worker comparison, default inspector reads, React update). Frame opportunity adds two animation-frame callbacks; it is not measured display latency and excludes touch hardware and OS input latency.' } };
    }));

    checks.push(await runCheck('ui-baseline-reset', async () => {
      const reset = button(frame.doc, 'Reset to baseline');
      if (!reset) throw new Error('Reset control is missing.');
      const computing = waitFor(frame, () => statusText(frame.doc).startsWith('Computing'), 'reset to start');
      reset.click(); await computing;
      await waitFor(frame, () => settled(frame.doc) && statusText(frame.doc) === 'Showing 1.00×', 'the baseline');
      const strategy = frame.doc.querySelector('[data-testid="execution-strategy"]')?.textContent;
      const deltas = [...frame.doc.querySelectorAll('.probability-row .delta')].map(element => element.textContent);
      if (strategy !== 'Baseline reused') throw new Error(`Reset used "${strategy}" instead of the immutable baseline.`);
      if (!deltas.length || deltas.some(delta => delta !== '+0.00 pp')) throw new Error(`Reset left displayed changes: ${deltas.join(', ')}`);
      return { data: { strategy, displayedRows: deltas.length } };
    }));

    checks.push(await runCheck('ui-cpu-recovery', async () => {
      const restart = button(frame.doc, 'Restart computation worker on CPU');
      if (!restart) throw new Error('Restart control is missing.');
      const loading = waitFor(frame, () => statusText(frame.doc).startsWith('Loading'), 'the replacement worker to start');
      const started = frame.win.performance.now();
      restart.click(); await loading;
      const ready = await waitFor(frame, () => settled(frame.doc) || !!frame.doc.querySelector('[role="alert"]'), 'the replacement worker result');
      const alert = frame.doc.querySelector('[role="alert"]')?.textContent;
      if (alert) throw new Error(alert);
      if (select(frame.doc, 'Backend').value !== 'cpu') throw new Error('The replacement worker is not using CPU.');
      if (statusText(frame.doc) !== 'Showing 1.00×') throw new Error(`Restart did not preserve the requested strength: ${statusText(frame.doc)}.`);
      if (frame.errors.length) throw new Error(`Page errors: ${frame.errors.join('; ')}`);
      return { detail: `${(ready.at - started).toFixed(1)} ms`, data: { recoveryMs: ready.at - started, definition: 'Restart click until the replacement CPU worker shows the preserved experiment.' } };
    }));
    return checks;
  } finally { host.replaceChildren(); }
}

// ---- Worker-level numerical and computation checks ----

function benchmarkInput(model: HarnessModel, loaded: Loaded, recipeText: string | undefined) {
  return tokenize(model === 'retrieval' && recipeText ? recipeText : verificationInput, loaded.tokenizer, loaded.manifest.config.contextLength);
}

async function workerChecks(model: HarnessModel, backend: HarnessBackend, settings: HarnessSettings, base: URL, stamp: ReleaseStamp | null,
  record: CombinationRecord, progress: Progress): Promise<CheckRecord[]> {
  const manifestUrl = new URL(`models/${model}/manifest.json`, base).href;
  const client = new ModelClient();
  const checks: CheckRecord[] = [];
  let loaded: Loaded | null = null;
  try {
    progress(`${model}/${backend}: loading a fresh worker`);
    checks.push(await runCheck('worker-load', async () => {
      const started = performance.now();
      try { loaded = await client.request('LOAD', { manifestUrl, backend }); }
      catch (error) {
        if (error instanceof EngineError && error.code === 'UNSUPPORTED_BACKEND') return { status: 'unsupported', detail: error.message };
        throw error;
      }
      const result = loaded!;
      record.actualBackend = result.backend; record.modelId = result.manifest.modelId; record.modelRevision = result.manifest.modelRevision; record.precision = result.precision;
      if (result.backend !== backend) throw new Error(`Requested ${backend}; the worker reported ${result.backend}.`);
      const stamped = stamp?.models?.[model]?.modelRevision;
      if (stamped && stamped !== result.manifest.modelRevision) throw new Error(`Loaded revision ${result.manifest.modelRevision} differs from release stamp ${stamped}.`);
      return { data: { loadMs: performance.now() - started, fallbackReason: result.fallbackReason ?? null, matchesReleaseStamp: stamped ? true : null,
        definition: 'Worker LOAD request: static asset fetch, hash validation, backend initialization, and weight upload.' } };
    }));
    const pending = ['computation-rpc', 'reference-probe', 'prepared-recipes', 'replay-roundtrip', 'resource-cleanup', 'webgl-context-loss'];
    if (checks[0].status !== 'passed' || !loaded) {
      for (const name of pending) checks.push({ name, status: 'skipped', detail: checks[0].status === 'unsupported' ? 'Backend unsupported.' : 'The worker did not load.' });
      return checks;
    }
    const model0: Loaded = loaded;
    const config = model0.manifest.config;
    let catalog: ReturnType<typeof validateExperiments> | null = null;
    if (model === 'retrieval') {
      try { catalog = validateExperiments(await fetchExperiments(manifestUrl), model0.manifest, model0.tokenizer); }
      catch (error) { progress(`Prepared experiments could not be loaded: ${message(error)}`); }
    }

    checks.push(await runCheck('computation-rpc', async () => {
      progress(`${model}/${backend}: timing ${settings.measuredSamples} full/cached comparison pairs per layer`);
      const tokens = benchmarkInput(model, model0, catalog?.recipes[0]?.text);
      const started = performance.now();
      const baseline = await client.request('BASELINE', { tokenIds: tokens });
      const baselineFirstMs = performance.now() - started;
      const layers = [];
      try {
        for (const layer of [0, config.layers - 1]) {
          const samples = { full: [] as { rpcMs: number; workerMs: number }[], cached: [] as { rpcMs: number; workerMs: number }[] };
          for (let i = 0; i < settings.warmupSamples + settings.measuredSamples; i++) for (const mode of (i % 2 ? ['cached', 'full'] : ['full', 'cached']) as ('full' | 'cached')[]) {
            const begin = performance.now();
            const result = await client.request('COMPARE', { baselineRunId: baseline.runId, predictionPosition: tokens.length - 1,
              interventions: [{ id: 'benchmark', kind: 'head.scale', layer, head: benchmarkHead, factor: rpcFactor, tokens: { kind: 'all' } }],
              executionPreference: mode === 'full' ? 'full' : 'auto' });
            const rpcMs = performance.now() - begin;
            await client.request('RELEASE', { runId: result.modifiedRunId });
            if (result.execution.strategy !== (mode === 'full' ? 'full' : 'cached-suffix')) throw new Error(`Expected ${mode} execution; received ${result.execution.strategy}.`);
            if (i >= settings.warmupSamples) samples[mode].push({ rpcMs, workerMs: result.execution.elapsedMs });
          }
          layers.push({ layer, fullRoundTrip: summarize(samples.full.map(s => s.rpcMs)), cachedRoundTrip: summarize(samples.cached.map(s => s.rpcMs)),
            fullWorkerComputation: summarize(samples.full.map(s => s.workerMs)), cachedWorkerComputation: summarize(samples.cached.map(s => s.workerMs)), samples });
        }
      } finally { await client.request('RELEASE', { runId: baseline.runId }); }
      return { detail: layers.map(item => `layer ${item.layer + 1}: full/cached round-trip median ${item.fullRoundTrip.medianMs.toFixed(1)}/${item.cachedRoundTrip.medianMs.toFixed(1)} ms`).join('; '),
        data: { sequenceLength: tokens.length, baselineFirstMs, layers,
          definition: 'Round trip: postMessage through comparison response (transit, forward pass, logit readback, metrics). Worker computation: the engine-reported comparison time. Inspection reads and rendering are excluded.' } };
    }));

    checks.push(await runCheck('reference-probe', async () => {
      progress(`${model}/${backend}: comparing forward values, gradients, and Adam with PyTorch`);
      const probe = await client.request('PROBE', {});
      if (probe.backend !== backend) throw new Error(`Probe ran on ${probe.backend}.`);
      if (probe.retainedTensorDelta !== 0) throw new Error(`Probe retained ${probe.retainedTensorDelta} tensors.`);
      return { detail: `forward ${probe.forwardMaximumError.toExponential(2)}, gradient ${probe.gradientMaximumError.toExponential(2)}, Adam ${probe.optimizerMaximumError.toExponential(2)}`, data: probe };
    }));

    checks.push(await runCheck('prepared-recipes', async () => {
      if (model !== 'retrieval') return { status: 'skipped', detail: 'This checkpoint has no prepared recipes.' };
      if (!catalog) throw new Error('The prepared recipes could not be loaded or validated.');
      progress(`${model}/${backend}: reproducing prepared recipes`);
      const cases = [], problems: string[] = [];
      for (const recipe of catalog.recipes) {
        const baseline = await client.request('BASELINE', { tokenIds: recipe.tokenIds });
        try {
          const read = async (runId: string) => (await client.request('INSPECT', { runId, reference: { stage: 'logits' }, selection: { position: recipe.predictionPosition } })).values;
          const base0 = compareLogits(await read(baseline.runId), recipe.metrics.baselineLogits);
          const payload = { baselineRunId: baseline.runId, interventions: [recipe.intervention], predictionPosition: recipe.predictionPosition, targetTokenId: recipe.targetTokenId };
          const cached = await client.request('COMPARE', payload);
          const cachedLogits = compareLogits(await read(cached.modifiedRunId), recipe.metrics.modifiedLogits);
          await client.request('RELEASE', { runId: cached.modifiedRunId });
          const full = await client.request('COMPARE', { ...payload, executionPreference: 'full' });
          const fullLogits = compareLogits(await read(full.modifiedRunId), recipe.metrics.modifiedLogits);
          await client.request('RELEASE', { runId: full.modifiedRunId });
          const expectation = recipeExpectations[recipe.id];
          const item = { id: recipe.id, title: recipe.title, baselineMaximumError: base0.maximumError, cachedMaximumError: cachedLogits.maximumError, fullMaximumError: fullLogits.maximumError,
            strategy: cached.execution.strategy, hasComputationalPath: cached.hasComputationalPath, metrics: cached.metrics, fullMetrics: full.metrics,
            expectation: expectation?.describe ?? null, expectationHolds: expectation ? expectation.holds(cached.metrics) : null };
          cases.push(item);
          if (![base0, cachedLogits, fullLogits].every(result => result.withinTolerance)) problems.push(`${recipe.id}: logits outside 2e-5 + 1e-5|expected|`);
          if (cached.execution.strategy !== 'cached-suffix') problems.push(`${recipe.id}: used ${cached.execution.strategy}`);
          if (!expectation) problems.push(`${recipe.id}: no recorded expectation`);
          else {
            if (cached.hasComputationalPath !== expectation.reachable) problems.push(`${recipe.id}: reachability ${cached.hasComputationalPath}`);
            if (!item.expectationHolds) problems.push(`${recipe.id}: expected ${expectation.describe}`);
          }
        } finally { await client.request('RELEASE', { runId: baseline.runId }); }
      }
      if (problems.length) return { status: 'failed', detail: problems.join('; '), data: { cases } };
      return { detail: `${cases.length} recipes, maximum logit error ${Math.max(...cases.flatMap(c => [c.baselineMaximumError, c.cachedMaximumError, c.fullMaximumError])).toExponential(2)}`, data: { cases } };
    }));

    checks.push(await runCheck('replay-roundtrip', async () => {
      const recipe = catalog?.recipes[0];
      const tokens = benchmarkInput(model, model0, recipe?.text);
      const intervention: HeadScaleIntervention = recipe?.intervention ?? { id: 'head-strength', kind: 'head.scale', layer: 0, head: 0, factor: .5, tokens: { kind: 'all' } };
      const observation = { predictionPosition: recipe?.predictionPosition ?? tokens.length - 1, targetTokenId: recipe?.targetTokenId ?? 0, interventions: [intervention] };
      const runIds: string[] = [];
      const logits = async (runId: string) => (await client.request('INSPECT', { runId, reference: { stage: 'logits' }, selection: { position: observation.predictionPosition } })).values;
      try {
        const original = await client.request('BASELINE', { tokenIds: tokens }); runIds.push(original.runId);
        const compared = await client.request('COMPARE', { baselineRunId: original.runId, ...observation }); runIds.push(compared.modifiedRunId);
        const saved = createExperiment(model0.manifest, model0.tokenizer, tokens, observation, await logits(original.runId), await logits(compared.modifiedRunId),
          { backend: model0.backend, precision: model0.precision });
        const serialized = JSON.stringify(saved, null, 2) + '\n';
        const restored = parseExperiment(serialized);
        validateExperiment(restored, model0.manifest, model0.tokenizer);
        const replayBase = await client.request('BASELINE', { tokenIds: restored.input.tokenIds }); runIds.push(replayBase.runId);
        const replayed = await client.request('COMPARE', { baselineRunId: replayBase.runId, ...restored.observation }); runIds.push(replayed.modifiedRunId);
        const report = checkReplay(restored, await logits(replayBase.runId), await logits(replayed.modifiedRunId), model0.backend);
        if (!report.matches) return { status: 'failed', detail: 'Replayed logits differ from the saved experiment.', data: report };
        return { detail: `maximum difference ${Math.max(report.baselineMaximumError, report.modifiedMaximumError).toExponential(2)}`, data: { ...report, bytes: serialized.length,
          scope: 'In-memory export, JSON parse, validation, and recomputation. File download/import is covered by the manual protocol.' } };
      } finally { for (const runId of runIds) await client.request('RELEASE', { runId }).catch(() => {}); }
    }));

    checks.push(await runCheck('resource-cleanup', async () => {
      const resources = await client.request('RESOURCES', {});
      const expected = model0.manifest.tensors.length;
      if (resources.tensorCount !== expected || resources.cache.logicalBytes !== 0)
        throw new Error(`After releasing every run: ${resources.tensorCount} tensors (expected ${expected}), ${resources.cache.logicalBytes} cache bytes.`);
      return { data: resources };
    }));
    client.dispose();
    checks.push(await runCheck('webgl-context-loss', async () => {
      if (backend !== 'webgl') return { status: 'skipped', detail: 'Applies to WebGL only.' };
      progress(`${model}/${backend}: dropping the WebGL context and recovering on CPU`);
      return contextLossCheck(manifestUrl, benchmarkInput(model, model0, catalog?.recipes[0]?.text));
    }));
    return checks;
  } finally { client.dispose(); }
}

/** Uses WEBGL_lose_context, so detection follows the browser's own context-loss event, then recovers on a fresh CPU worker. */
async function contextLossCheck(manifestUrl: string, tokens: number[]): Promise<{ status?: 'unsupported' | 'failed'; detail: string; data?: unknown }> {
  const lost = new ModelClient();
  let lossToEndMs: number | null = null, afterLoss: string;
  try {
    await lost.request('LOAD', { manifestUrl, backend: 'webgl' });
    const baseline = await lost.request('BASELINE', { tokenIds: tokens });
    const ended = new Promise<void>(resolve => { lost.onEnded = () => resolve(); });
    const started = performance.now();
    try { await lost.request('LOSE_CONTEXT', {}); }
    catch (error) {
      if (error instanceof EngineError && error.code === 'UNSUPPORTED_BACKEND') return { status: 'unsupported', detail: error.message };
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const noticed = await Promise.race([ended.then(() => true), new Promise<boolean>(resolve => { timer = setTimeout(() => resolve(false), 10_000); })]);
    clearTimeout(timer);
    if (noticed) lossToEndMs = performance.now() - started;
    afterLoss = await lost.request('COMPARE', { baselineRunId: baseline.runId, predictionPosition: tokens.length - 1,
      interventions: [{ id: 'context-loss', kind: 'head.scale', layer: 0, head: benchmarkHead, factor: rpcFactor, tokens: { kind: 'all' } }] })
      .then(() => 'returned a result', error => `rejected: ${message(error)}`);
    if (afterLoss === 'returned a result') return { status: 'failed', detail: 'A comparison returned values after the context was lost.' };
    if (!noticed) return { status: 'failed', detail: `No context-loss event ended the session within 10 s; the next comparison ${afterLoss}.` };
  } finally { lost.dispose(); }
  const recovery = new ModelClient();
  try {
    const started = performance.now();
    const loaded = await recovery.request('LOAD', { manifestUrl, backend: 'cpu' });
    const probe = await recovery.request('PROBE', {});
    if (loaded.backend !== 'cpu' || probe.retainedTensorDelta !== 0) throw new Error('The replacement CPU worker did not pass its reference probe.');
    const recoveryMs = performance.now() - started;
    return { detail: `session ended ${lossToEndMs!.toFixed(1)} ms after the loss; CPU worker verified in ${recoveryMs.toFixed(1)} ms`,
      data: { lossToSessionEndMs: lossToEndMs, afterLoss, recoveryMs, probe,
        definition: 'WEBGL_lose_context request until the client ends the session from the worker notice; then a fresh CPU worker load and PyTorch probe.' } };
  } finally { recovery.dispose(); }
}

export interface HarnessEvidence { evidenceClass: EvidenceClass; deviceDescription: string; notes: string }

export async function runHarness(settings: HarnessSettings, evidence: HarnessEvidence, host: HTMLElement, progress: Progress,
  base = new URL('./', window.location.href)): Promise<ValidationReport> {
  let pageHiddenDuringRun = document.visibilityState === 'hidden';
  const onVisibility = () => { if (document.visibilityState === 'hidden') pageHiddenDuringRun = true; };
  document.addEventListener('visibilitychange', onVisibility);
  let wakeLock: { release: () => Promise<void> } | null = null, screenWakeLock: ValidationReport['evidence']['screenWakeLock'] = 'unavailable';
  const wakeLockApi = (navigator as Navigator & { wakeLock?: { request: (kind: 'screen') => Promise<{ release: () => Promise<void> }> } }).wakeLock;
  if (wakeLockApi) { try { wakeLock = await wakeLockApi.request('screen'); screenWakeLock = 'acquired'; } catch { screenWakeLock = 'denied'; } }
  const report: ValidationReport = {
    format: 'transformer-microscope-device-validation', schemaVersion: 1, startedAt: new Date().toISOString(), completedAt: null,
    evidence: { ...evidence, automated: typeof navigator.webdriver === 'boolean' ? navigator.webdriver : null, pageHiddenDuringRun, screenWakeLock },
    page: { url: window.location.href, crossOriginIsolated: self.crossOriginIsolated ?? false },
    release: { name: 'release-identity', status: 'skipped' }, environment: {},
    settings: { ...settings, warmFactors, head: benchmarkHead, rpcFactor }, combinations: [],
    summary: { passed: 0, failed: 0, unsupported: 0, skipped: 0, outcome: 'incomplete' },
  };
  try {
    progress('Recording browser and release identity');
    report.environment = await captureEnvironment();
    const release = await readRelease(base);
    report.release = release.record;
    for (const model of settings.models) for (const backend of settings.backends) {
      const record: CombinationRecord = { model, requestedBackend: backend, actualBackend: null, modelId: null, modelRevision: null, precision: null, checks: [] };
      report.combinations.push(record);
      if (settings.includeInterface) record.checks.push(...await interfaceChecks(host, model, backend, settings, base, progress));
      record.checks.push(...await workerChecks(model, backend, settings, base, release.stamp, record, progress));
    }
    report.completedAt = new Date().toISOString();
  } catch (error) {
    // Individual checks record their own failures; this is an unexpected harness fault, so the run stays incomplete.
    report.abortReason = message(error); progress(`Harness stopped: ${report.abortReason}`);
  } finally {
    document.removeEventListener('visibilitychange', onVisibility);
    await wakeLock?.release().catch(() => {});
    report.evidence.pageHiddenDuringRun = pageHiddenDuringRun;
    report.summary = summarizeChecks(report.release, report.combinations, report.completedAt !== null);
  }
  return report;
}
