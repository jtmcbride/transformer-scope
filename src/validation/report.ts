/** Pure report structures for the opt-in device validation harness. */

export type CheckStatus = 'passed' | 'failed' | 'unsupported' | 'skipped';
export interface CheckRecord { name: string; status: CheckStatus; detail?: string; durationMs?: number; data?: unknown }
export type HarnessModel = 'retrieval' | 'verification';
export type HarnessBackend = 'cpu' | 'webgl';
export type EvidenceClass = 'unspecified' | 'physical-device' | 'installed-desktop-browser' | 'viewport-emulation' | 'automation';
export const evidenceClasses: readonly EvidenceClass[] = ['unspecified', 'physical-device', 'installed-desktop-browser', 'viewport-emulation', 'automation'];

export interface HarnessSettings {
  models: HarnessModel[]; backends: HarnessBackend[];
  warmupSamples: number; measuredSamples: number; includeInterface: boolean;
}
/** The same warm-edit methodology as tests/benchmarks/browser.benchmark.ts. */
export const warmFactors = [.25, .5, .75, 1.5] as const;
export const benchmarkHead = 0;
export const rpcFactor = .5;
export const defaultSettings: HarnessSettings = { models: ['retrieval', 'verification'], backends: ['cpu', 'webgl'], warmupSamples: 5, measuredSamples: 40, includeInterface: true };

export interface CombinationRecord {
  model: HarnessModel; requestedBackend: HarnessBackend; actualBackend: string | null;
  modelId: string | null; modelRevision: string | null; precision: unknown; checks: CheckRecord[];
}
export interface ValidationReport {
  format: 'transformer-microscope-device-validation'; schemaVersion: 1;
  startedAt: string; completedAt: string | null; abortReason?: string;
  evidence: { evidenceClass: EvidenceClass; deviceDescription: string; notes: string; automated: boolean | null;
    pageHiddenDuringRun: boolean; screenWakeLock: 'acquired' | 'unavailable' | 'denied' };
  page: { url: string; crossOriginIsolated: boolean };
  release: CheckRecord; environment: Record<string, unknown>; settings: HarnessSettings & { warmFactors: readonly number[]; head: number; rpcFactor: number };
  combinations: CombinationRecord[];
  summary: { passed: number; failed: number; unsupported: number; skipped: number; outcome: 'passed' | 'failed' | 'incomplete' };
}

export interface Distribution { count: number; medianMs: number; p95Ms: number; minimumMs: number; maximumMs: number }
/** Median and nearest-rank p95, matching the browser benchmark summaries. */
export function summarize(values: readonly number[]): Distribution {
  if (!values.length || values.some(value => !Number.isFinite(value) || value < 0)) throw new Error('Timing samples must be finite and non-negative.');
  const sorted = [...values].sort((a, b) => a - b);
  return { count: sorted.length, medianMs: (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * .95) - 1], minimumMs: sorted[0], maximumMs: sorted[sorted.length - 1] };
}

/** Reference logit comparison with the existing replay/recipe bound: |a - e| <= 2e-5 + 1e-5 |e|. */
export function compareLogits(actual: ArrayLike<number>, expected: ArrayLike<number>): { maximumError: number; withinTolerance: boolean } {
  if (actual.length !== expected.length) throw new Error(`Logit length mismatch: ${actual.length} versus ${expected.length}.`);
  let maximumError = 0, withinTolerance = true;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) return { maximumError: Infinity, withinTolerance: false };
    const error = Math.abs(actual[i] - expected[i]);
    maximumError = Math.max(maximumError, error);
    if (error > 2e-5 + 1e-5 * Math.abs(expected[i])) withinTolerance = false;
  }
  return { maximumError, withinTolerance };
}

/** Expected recipe behavior. These are bounded checks on the shipped examples, not head labels. */
export const recipeExpectations: Record<string, { reachable: boolean; describe: string; holds: (metrics: { targetLogProbabilityDelta?: number; jsDivergence: number; maxAbsoluteLogitChange: number }) => boolean }> = {
  effect: { reachable: true, describe: 'target Δ log p < -1 nat', holds: metrics => (metrics.targetLogProbabilityDelta ?? 0) < -1 },
  small: { reachable: true, describe: 'JSD < 0.001 nats', holds: metrics => metrics.jsDivergence < .001 },
  unreachable: { reachable: false, describe: 'maximum logit change exactly 0', holds: metrics => metrics.maxAbsoluteLogitChange === 0 },
};

export function validateSettings(settings: HarnessSettings): HarnessSettings {
  const integer = (value: number, minimum: number, maximum: number, name: string) => {
    if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} to ${maximum}.`);
  };
  if (!settings.models.length || settings.models.some(model => model !== 'retrieval' && model !== 'verification')) throw new Error('Choose at least one checkpoint.');
  if (!settings.backends.length || settings.backends.some(backend => backend !== 'cpu' && backend !== 'webgl')) throw new Error('Choose at least one backend.');
  integer(settings.warmupSamples, 0, 50, 'Warm-up edits'); integer(settings.measuredSamples, 1, 200, 'Measured edits');
  return { ...settings, models: [...new Set(settings.models)], backends: [...new Set(settings.backends)] };
}

export function summarizeChecks(release: CheckRecord, combinations: readonly CombinationRecord[], completed: boolean): ValidationReport['summary'] {
  const counts = { passed: 0, failed: 0, unsupported: 0, skipped: 0 };
  for (const check of [release, ...combinations.flatMap(combination => combination.checks)]) counts[check.status]++;
  // CPU is the required fallback; it can never be "unsupported" for a passing run.
  const cpuMissing = combinations.some(combination => combination.requestedBackend === 'cpu' && combination.checks.some(check => check.status === 'unsupported'));
  return { ...counts, outcome: !completed ? 'incomplete' : counts.failed || cpuMissing ? 'failed' : 'passed' };
}

/** A failure inside one check is recorded; it does not abort independent checks. */
export async function runCheck(name: string, work: () => Promise<{ status?: CheckStatus; detail?: string; data?: unknown } | void>): Promise<CheckRecord> {
  const started = performance.now();
  try {
    const result = await work() ?? {};
    return { name, status: result.status ?? 'passed', ...(result.detail ? { detail: result.detail } : {}), durationMs: performance.now() - started, ...(result.data !== undefined ? { data: result.data } : {}) };
  } catch (error) {
    return { name, status: 'failed', detail: error instanceof Error ? error.message : String(error), durationMs: performance.now() - started };
  }
}
