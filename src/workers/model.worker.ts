import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import { EngineError, type ModelArtifact } from '../engine/contracts';
import { loadArtifact } from '../engine/weights';
import { RunEngine } from '../engine/runs';
import { runProbe, type ProbeFixture } from '../engine/probe';
import type { WorkerRequest, WorkerResponse } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;
let engine: RunEngine | undefined;
let artifact: ModelArtifact | undefined;
let manifestUrl = '';
let queue = Promise.resolve();

async function load(request: Extract<WorkerRequest, { command: 'LOAD' }>) {
  engine?.dispose(); engine = undefined; artifact = undefined;
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
      // Force a kernel and a readback before accepting the backend.
      const probe = tf.tidy(() => tf.add(tf.tensor1d([1]), tf.tensor1d([2])));
      try { await probe.data(); } finally { probe.dispose(); }
    } catch (error) {
      if (desired === 'webgl') throw new EngineError('UNSUPPORTED_BACKEND', `WebGL worker unavailable: ${String(error)}`);
      fallbackReason = 'WebGL is unavailable in this worker. Using CPU.';
      await tf.setBackend('cpu'); await tf.ready();
    }
  } else { await tf.setBackend('cpu'); await tf.ready(); }
  artifact = nextArtifact;
  engine = new RunEngine(artifact, { cacheBudgetBytes: request.payload.cacheBudgetBytes });
  return { manifest: artifact.manifest, tokenizer: artifact.tokenizer, backend: tf.getBackend(), fallbackReason,
    runtimeGeneration: engine.runtimeGeneration, precision: { dtype: 'float32',
      float32Textures: tf.getBackend() === 'webgl' ? tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') : null,
      webglPacking: tf.getBackend() === 'webgl' ? tf.env().getBool('WEBGL_PACK') : null } };
}

async function execute(request: WorkerRequest): Promise<unknown> {
  if (request.command === 'LOAD') return load(request);
  if (!engine || !artifact) throw new EngineError('STALE_SESSION', 'Load a model before running an experiment.');
  switch (request.command) {
    case 'BASELINE': return engine.runBaseline(request.payload.tokenIds);
    case 'COMPARE': return engine.compare(request.payload);
    case 'INSPECT': return engine.inspectActivation(request.payload.runId, request.payload.reference, request.payload.selection);
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
    const transfers: Transferable[] = response.ok && request.command === 'INSPECT'
      ? [(response.result as { values: Float32Array }).values.buffer] : [];
    scope.postMessage(response, transfers);
  });
};
