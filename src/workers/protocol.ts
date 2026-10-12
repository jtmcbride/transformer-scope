import type { ActivationReference, EngineErrorCode, ModelManifest, SymbolTokenizer } from '../engine/contracts';
import type { ActivationSlice, CompareRequest, ComparisonResult, RunSummary } from '../engine/runs';
import type { ProbeReport } from '../engine/probe';

export interface Commands {
  LOAD: { request: { manifestUrl: string; backend: 'cpu' | 'webgl' | 'auto' }; response: {
    manifest: ModelManifest; tokenizer: SymbolTokenizer; backend: string; fallbackReason?: string; runtimeGeneration: string } };
  BASELINE: { request: { tokenIds: number[] }; response: RunSummary };
  COMPARE: { request: CompareRequest; response: ComparisonResult };
  INSPECT: { request: { runId: string; reference: ActivationReference; selection?: Record<string, number> }; response: ActivationSlice };
  RELEASE: { request: { runId: string }; response: null };
  PROBE: { request: Record<string, never>; response: ProbeReport };
}
export type Command = keyof Commands;
export type WorkerRequest = { [K in Command]: { id: number; command: K; payload: Commands[K]['request'] } }[Command];
export type WorkerResponse = { id: number; ok: true; result: unknown } | { id: number; ok: false; error: { code: EngineErrorCode; message: string } };
