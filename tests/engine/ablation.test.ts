import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import { comparePredictions } from '../../src/engine/comparison';
import type { HeadScaleIntervention } from '../../src/engine/contracts';
import fixture from '../fixtures/ablation-reference.json';
import { artifact, manifest, tokenizer } from './retrieval-artifact';
import { assertClose } from './numerical';

const ablate = (layer: number, head: number): HeadScaleIntervention => ({ id: `ablate-${layer}-${head}`, kind: 'head.scale', layer, head, factor: 0, tokens: { kind: 'all' } });
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });

it('scans every single-head ablation and matches independent PyTorch logits', async () => {
  expect(fixture.modelRevision).toBe(manifest.modelRevision);
  const start = tf.memory().numTensors;
  const engine = new RunEngine(artifact);
  let maximum = 0;
  try {
    for (const input of fixture.inputs) {
      const baseline = await engine.runBaseline(input.tokenIds);
      const scan = await engine.scanHeadAblations(baseline.runId, input.predictionPosition, input.targetTokenId);
      maximum = Math.max(maximum, assertClose(scan.baselineLogits, input.baselineLogits, `${input.text} baseline`));
      expect(scan.entries.map(entry => [entry.layer, entry.head])).toEqual(input.single.map(entry => [entry.layer, entry.head]));
      for (const [i, entry] of scan.entries.entries()) {
        maximum = Math.max(maximum, assertClose(entry.logits, input.single[i].logits, `${input.text} L${entry.layer + 1}H${entry.head + 1}`));
        // Metrics come from the same comparison code applied to the returned logits.
        const recomputed = comparePredictions(scan.baselineLogits, entry.logits, tokenizer, 1, input.targetTokenId);
        expect(entry.metrics).toEqual(recomputed.metrics);
        expect(entry.topTokenId).toBe(recomputed.modifiedTopTokens[0].tokenId);
        expect(entry.strategy).toBe('cached-suffix');
      }
      for (const pair of input.pairs) {
        const result = await engine.compare({ baselineRunId: baseline.runId, predictionPosition: input.predictionPosition, targetTokenId: input.targetTokenId,
          interventions: pair.heads.map(([layer, head]) => ablate(layer, head)) });
        const logits = await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position: input.predictionPosition });
        maximum = Math.max(maximum, assertClose(logits.values, pair.logits, `${input.text} pair ${JSON.stringify(pair.heads)}`));
        expect(result.execution.strategy).toBe(pair.heads[0][0] === pair.heads[1][0] ? 'cached-suffix' : 'full');
        engine.releaseRun(result.modifiedRunId);
      }
      engine.releaseRun(baseline.runId);
    }
    await expect(engine.scanHeadAblations('missing', 0, 0)).rejects.toThrow();
  } finally { engine.dispose(); }
  console.info('Maximum ablation logit error', maximum);
  expect(tf.memory().numTensors).toBe(start);
});

it('measures non-additive pairs: two heads with negligible single effects can matter together', async () => {
  const engine = new RunEngine(artifact);
  try {
    const input = fixture.inputs[1];
    const baseline = await engine.runBaseline(input.tokenIds);
    const scan = await engine.scanHeadAblations(baseline.runId, input.predictionPosition, input.targetTokenId);
    const single = (layer: number, head: number) => scan.entries.find(entry => entry.layer === layer && entry.head === head)!.metrics.targetLogProbabilityDelta!;
    const joint = await engine.compare({ baselineRunId: baseline.runId, predictionPosition: input.predictionPosition, targetTokenId: input.targetTokenId, interventions: [ablate(0, 1), ablate(0, 3)] });
    expect(single(0, 1) + single(0, 3)).toBeGreaterThan(-.01);
    expect(joint.metrics.targetLogProbabilityDelta!).toBeLessThan(-5);
  } finally { engine.dispose(); }
});
