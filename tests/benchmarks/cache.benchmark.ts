import { mkdirSync, writeFileSync } from 'node:fs';
import { test } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import fixture from '../fixtures/suffix-reference.json';
import { TransformerModel, disposeForward, type ForwardResult } from '../../src/engine/model';
import type { ActivationReference, HeadScaleIntervention } from '../../src/engine/contracts';

// Explicitly invoked measurement command; timings are observations, not test thresholds.
test('records full-versus-suffix compute/readback latency for early, middle, and late edits', async () => {
  await tf.setBackend('cpu'); await tf.ready();
  const reference = fixture.models[0];
  const model = new TransformerModel(reference.config, reference.weights);
  const tokens = [1, 5, 9, 3, 7, 2, 11, 4];
  const capture: ActivationReference[] = Array.from({ length: reference.config.layers }, (_, layer) => [
    { layer, stage: 'attention.head.raw' as const }, { layer, stage: 'attention.residual' as const },
  ]).flat();
  const baseline = model.forward(tokens, [], capture);
  const results = [];
  function summary(values: number[]) {
    const sorted = [...values].sort((a, b) => a - b);
    return { medianMs: (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2,
      p95Ms: sorted[Math.ceil(sorted.length * .95) - 1] };
  }
  async function measure(run: () => ForwardResult): Promise<number> {
    const start = performance.now();
    const result = run();
    try { await result.logits.data(); return performance.now() - start; }
    finally { disposeForward(result); }
  }
  try {
    for (let layer = 0; layer < reference.config.layers; layer++) {
      const patch: HeadScaleIntervention = { id: 'benchmark', kind: 'head.scale', layer, head: 0, factor: .5, tokens: { kind: 'all' } };
      const cache = { rawHeadOutputs: baseline.captured[`${layer}:attention.head.raw`] as tf.Tensor4D,
        attentionResidual: baseline.captured[`${layer}:attention.residual`] as tf.Tensor3D };
      const full = () => model.forward(tokens, [patch]);
      const cached = () => model.forwardCached(cache, layer, [patch]);
      for (let warmup = 0; warmup < 10; warmup++) { await measure(full); await measure(cached); }
      const fullTimes: number[] = [], cachedTimes: number[] = [];
      for (let sample = 0; sample < 80; sample++) {
        if (sample % 2) { cachedTimes.push(await measure(cached)); fullTimes.push(await measure(full)); }
        else { fullTimes.push(await measure(full)); cachedTimes.push(await measure(cached)); }
      }
      const fullResult = summary(fullTimes), cachedResult = summary(cachedTimes);
      results.push({ layer, full: fullResult, cached: cachedResult, medianSpeedup: fullResult.medianMs / cachedResult.medianMs });
    }
  } finally { disposeForward(baseline); model.dispose(); }
  const report = { timestampUTC: new Date().toISOString(), nodeVersion: process.version, platform: process.platform, architecture: process.arch,
    backend: tf.getBackend(), seed: reference.seed, config: reference.config, tokenIds: tokens,
    warmupPairs: 10, samplePairs: 80, scope: 'model forward plus all-logit readback; excludes worker, comparison metrics, inspection, rendering, and disposal time', results };
  mkdirSync('.cache', { recursive: true });
  writeFileSync('.cache/cache-benchmark.json', JSON.stringify(report, null, 2) + '\n');
  console.info(JSON.stringify(report));
});
