import type { ActivationReference, EngineErrorCode, ModelManifest, SymbolTokenizer } from '../engine/contracts';
import type { ActivationSlice, CompareRequest, ComparisonResult, RunSummary } from '../engine/runs';
import type { ProbeReport } from '../engine/probe';
import type { CacheStats } from '../engine/cache';

export interface Commands {
  LOAD: { request: { manifestUrl: string; backend: 'cpu' | 'webgl' | 'auto'; cacheBudgetBytes?: number }; response: {
    manifest: ModelManifest; tokenizer: SymbolTokenizer; backend: string; fallbackReason?: string; runtimeGeneration: string } };
  BASELINE: { request: { tokenIds: number[] }; response: RunSummary };
  COMPARE: { request: CompareRequest; response: ComparisonResult };
  INSPECT: { request: { runId: string; reference: ActivationReference; selection?: Record<string, number> }; response: ActivationSlice };
  RELEASE: { request: { runId: string }; response: null };
  PROBE: { request: Record<string, never>; response: ProbeReport };
  RESOURCES: { request: Record<string, never>; response: { cache: CacheStats; tensorCount: number; logicalTensorBytes: number; memoryEstimateUnreliable: boolean } };
}
export type Command = keyof Commands;
export type WorkerRequest = { [K in Command]: { id: number; command: K; payload: Commands[K]['request'] } }[Command];
export type WorkerResponse = { id: number; ok: true; result: unknown } | { id: number; ok: false; error: { code: EngineErrorCode; message: string } };
