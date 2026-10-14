import type { ActivationReference, EngineErrorCode, ModelManifest, ModelTokenizer } from '../engine/contracts';
import type { ActivationSlice, CompareRequest, ComparisonResult, EmbeddingSlice, HeadAblationScan, ResidualLens, RunSummary } from '../engine/runs';
import type { ProbeReport } from '../engine/probe';
import type { RuntimePrecision } from '../engine/experiment';
import type { CacheStats } from '../engine/cache';
import type { EffectTraceReport } from '../engine/trace';

export interface Commands {
  LOAD: { request: { manifestUrl: string; backend: 'cpu' | 'webgl' | 'auto'; cacheBudgetBytes?: number }; response: {
    manifest: ModelManifest; tokenizer: ModelTokenizer; backend: string; fallbackReason?: string; runtimeGeneration: string; precision: RuntimePrecision } };
  BASELINE: { request: { tokenIds: number[] }; response: RunSummary };
  COMPARE: { request: CompareRequest; response: ComparisonResult };
  INSPECT: { request: { runId: string; reference: ActivationReference; selection?: Record<string, number> }; response: ActivationSlice };
  LENS: { request: { runId: string; position: number }; response: ResidualLens };
  TRACE: { request: { baselineRunId: string; modifiedRunId: string; predictionPosition: number }; response: EffectTraceReport };
  EMBEDDING: { request: { runId: string; position: number }; response: EmbeddingSlice };
  ABLATION_SCAN: { request: { baselineRunId: string; predictionPosition: number; targetTokenId: number }; response: HeadAblationScan };
  RELEASE: { request: { runId: string }; response: null };
  PROBE: { request: Record<string, never>; response: ProbeReport };
  RESOURCES: { request: Record<string, never>; response: { cache: CacheStats; tensorCount: number; logicalTensorBytes: number; memoryEstimateUnreliable: boolean } };
  /** Diagnostic for the validation harness: asks the browser to drop the worker's WebGL context. */
  LOSE_CONTEXT: { request: Record<string, never>; response: null };
}
export type Command = keyof Commands;
export type WorkerRequest = { [K in Command]: { id: number; command: K; payload: Commands[K]['request'] } }[Command];
export type WorkerResponse = { id: number; ok: true; result: unknown } | { id: number; ok: false; error: { code: EngineErrorCode; message: string } };
/** Unsolicited: the worker can no longer compute, so the client must end the session. */
export interface WorkerNotice { notice: 'BACKEND_LOST'; message: string }
