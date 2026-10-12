import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import type { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgl';
import { EngineError, type ModelArtifact } from '../engine/contracts';
import { loadArtifact } from '../engine/weights';
import { RunEngine } from '../engine/runs';
import { runProbe, type ProbeFixture } from '../engine/probe';
import type { WorkerNotice, WorkerRequest, WorkerResponse } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;
let engine: RunEngine | undefined;
let artifact: ModelArtifact | undefined;
let manifestUrl = '';
let queue = Promise.resolve();
let context: WebGLRenderingContext | null = null;
let backendLost = false;
const lostMessage = 'The browser reclaimed the GPU (WebGL context lost). Results after the loss were discarded. Restart the computation worker on CPU to continue.';

// A lost context cannot be restored under the existing tensors: readbacks stall or return
// meaningless values. End this worker so no displayed number comes from a lost context.
function loseBackend() {
  if (backendLost) return;
  backendLost = true;
  scope.postMessage({ notice: 'BACKEND_LOST', message: lostMessage } satisfies WorkerNotice);
}
function assertBackend() {
  if (context?.isContextLost()) loseBackend();
  if (backendLost) throw new EngineError('STALE_SESSION', lostMessage);
}
function watchContext(next: WebGLRenderingContext | null) {
  (context?.canvas as EventTarget | undefined)?.removeEventListener('webglcontextlost', loseBackend);
  context = next;
  (context?.canvas as EventTarget | undefined)?.addEventListener('webglcontextlost', loseBackend);
}

async function load(request: Extract<WorkerRequest, { command: 'LOAD' }>) {
  engine?.dispose(); engine = undefined; artifact = undefined; watchContext(null);
  manifestUrl = request.payload.manifestUrl;
  const nextArtifact = await loadArtifact(manifestUrl);
  const parameterCount = nextArtifact.manifest.tensors.reduce((sum, tensor) => sum + tensor.byteLength / 4, 0);
  // Named-checkpoint measurements favor CPU startup and meet the warm interaction target.
  // Recalibrate this preference when weights/configuration change; explicit WebGL remains available.
  const measuredCpuRevision = 'd95b28530c047010f403dc6b8c7256c7d9223df2cf83a02bd4a5a1bcb9d64eea';
  const cpuPreferred = parameterCount < 10_000 || nextArtifact.manifest.modelRevision === measuredCpuRevision;
  const desired = request.payload.backend === 'auto' && cpuPreferred ? 'cpu' : request.payload.backend;
  let fallbackReason: string | undefined;
  if (desired !== 'cpu') {
    try {
      // Packed execution produced non-finite tied-embedding gradients on tested WebKit.
      // Keep the independently validated unpacked path until packing passes this graph.
      tf.env().set('WEBGL_PACK', false);
      if (!await tf.setBackend('webgl')) throw new Error('No worker WebGL context is available.');
      await tf.ready();
      if (!tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED')) throw new Error('This worker does not support float32 WebGL rendering.');
      // Force a kernel and a readback before accepting the backend.
      const probe = tf.tidy(() => tf.add(tf.tensor1d([1]), tf.tensor1d([2])));
      try { await probe.data(); } finally { probe.dispose(); }
    } catch (error) {
      if (desired === 'webgl') throw new EngineError('UNSUPPORTED_BACKEND', `WebGL worker unavailable: ${String(error)}`);
      fallbackReason = 'WebGL is unavailable in this worker. Using CPU.';
      await tf.setBackend('cpu'); await tf.ready();
    }
  } else { await tf.setBackend('cpu'); await tf.ready(); }
  if (tf.getBackend() === 'webgl') watchContext((tf.backend() as MathBackendWebGL).gpgpu.gl);
  artifact = nextArtifact;
  engine = new RunEngine(artifact, { cacheBudgetBytes: request.payload.cacheBudgetBytes });
  return { manifest: artifact.manifest, tokenizer: artifact.tokenizer, backend: tf.getBackend(), fallbackReason,
    runtimeGeneration: engine.runtimeGeneration, precision: { dtype: 'float32',
      float32Textures: tf.getBackend() === 'webgl' ? tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') : null,
      webglPacking: tf.getBackend() === 'webgl' ? tf.env().getBool('WEBGL_PACK') : null } };
}

async function execute(request: WorkerRequest): Promise<unknown> {
  assertBackend();
  if (request.command === 'LOSE_CONTEXT') {
    const extension = tf.getBackend() === 'webgl' ? context?.getExtension('WEBGL_lose_context') : null;
    if (!extension) throw new EngineError('UNSUPPORTED_BACKEND', 'Context-loss simulation requires WebGL with WEBGL_lose_context.');
    // Detection is left to the browser's webglcontextlost event, the same path as a real loss.
    extension.loseContext(); return null;
  }
  const result = await run(request);
  assertBackend();
  return result;
}

async function run(request: Exclude<WorkerRequest, { command: 'LOSE_CONTEXT' }>): Promise<unknown> {
  if (request.command === 'LOAD') return load(request);
  if (!engine || !artifact) throw new EngineError('STALE_SESSION', 'Load a model before running an experiment.');
  switch (request.command) {
    case 'BASELINE': return engine.runBaseline(request.payload.tokenIds);
    case 'COMPARE': return engine.compare(request.payload);
    case 'INSPECT': return engine.inspectActivation(request.payload.runId, request.payload.reference, request.payload.selection);
    case 'LENS': return engine.inspectResidualLens(request.payload.runId, request.payload.position);
    case 'EMBEDDING': return engine.inspectEmbedding(request.payload.runId, request.payload.position);
    case 'RELEASE': engine.releaseRun(request.payload.runId); return null;
    case 'RESOURCES': {
      const memory = tf.memory();
      return { cache: engine.cacheStats(), tensorCount: memory.numTensors, logicalTensorBytes: memory.numBytes, memoryEstimateUnreliable: memory.unreliable };
    }
    case 'PROBE': {
      const response = await fetch(new URL('probe.json', manifestUrl));
      if (!response.ok) throw new EngineError('INCOMPATIBLE_MODEL', 'The verification probe could not be loaded.');
      return runProbe(artifact, await response.json() as ProbeFixture);
    }
  }
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  // Model loading and resource release cannot run concurrently with an execution.
  queue = queue.then(async () => {
    let response: WorkerResponse;
    try { response = { id: request.id, ok: true, result: await execute(request) }; }
    catch (error) { response = { id: request.id, ok: false, error: {
      code: error instanceof EngineError ? error.code : 'NUMERICAL_FAILURE',
      message: error instanceof Error ? error.message : String(error),
    } }; }
    const transfers: Transferable[] = !response.ok ? []
      : request.command === 'INSPECT' ? [(response.result as { values: Float32Array }).values.buffer]
      : request.command === 'LENS' ? [(response.result as { norms: Float32Array }).norms.buffer, (response.result as { logits: Float32Array }).logits.buffer] : [];
    scope.postMessage(response, transfers);
  });
};
