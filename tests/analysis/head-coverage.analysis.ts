import { mkdirSync, writeFileSync } from 'node:fs';
import { cpus, platform, release } from 'node:os';
import { expect, test } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import type { HeadScaleIntervention } from '../../src/engine/contracts';
import { partition, retrievalExamples } from './retrieval-split';
import { artifact, manifest } from '../engine/retrieval-artifact';

// Explicitly invoked analysis (npm run analysis:heads). It measures single-head ablations on the
// held-out split with the verified engine; it does not assign heads semantic roles.

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1))];

test('single-head ablation effects across the held-out split', { timeout: 600_000 }, async () => {
  await tf.setBackend('cpu'); await tf.ready();
  const inputs = retrievalExamples().filter(example => partition(example) === 'test');
  expect(inputs).toHaveLength(453);
  const { layers, heads } = manifest.config;
  const configurations = Array.from({ length: layers }, (_, layer) => Array.from({ length: heads }, (_, head) =>
    (['all', 0] as const).map(scope => ({ layer, head, scope })))).flat(2);
  const effects = configurations.map(() => [] as { delta: number; jsd: number; maxLogitChange: number; changed: boolean; probability: number; reachable: boolean }[]);
  const strongest = new Map<string, number>();
  const engine = new RunEngine(artifact);
  try {
    for (const input of inputs) {
      const baseline = await engine.runBaseline(input.tokens);
      const position = input.tokens.length - 1;
      let worst = { key: '', delta: Infinity };
      for (const [index, configuration] of configurations.entries()) {
        const intervention: HeadScaleIntervention = { id: 'coverage', kind: 'head.scale', layer: configuration.layer, head: configuration.head, factor: 0,
          tokens: configuration.scope === 'all' ? { kind: 'all' } : { kind: 'indices', indices: [configuration.scope] } };
        const result = await engine.compare({ baselineRunId: baseline.runId, interventions: [intervention], predictionPosition: position, targetTokenId: input.target, topK: 1 });
        engine.releaseRun(result.modifiedRunId);
        const delta = result.metrics.targetLogProbabilityDelta!;
        const answer = result.modifiedTopTokens[0];
        effects[index].push({ delta, jsd: result.metrics.jsDivergence, maxLogitChange: result.metrics.maxAbsoluteLogitChange,
          changed: answer.tokenId !== input.target, probability: result.displayTokens.find(token => token.tokenId === input.target)!.probability, reachable: result.hasComputationalPath });
        if (configuration.scope === 'all' && delta < worst.delta) worst = { key: `L${configuration.layer + 1}H${configuration.head + 1}`, delta };
      }
      strongest.set(worst.key, (strongest.get(worst.key) ?? 0) + 1);
      engine.releaseRun(baseline.runId);
      expect(baseline.tokenIds).toEqual(input.tokens);
    }
  } finally { engine.dispose(); }
  const summaries = configurations.map((configuration, index) => {
    const items = effects[index], deltas = items.map(item => item.delta).sort((a, b) => a - b), jsd = items.map(item => item.jsd).sort((a, b) => a - b);
    return { layer: configuration.layer, head: configuration.head, label: `Layer ${configuration.layer + 1}, Head ${configuration.head + 1}`,
      positions: configuration.scope === 'all' ? 'all' : 'position 0 only', factor: 0, count: items.length,
      reachableCount: items.filter(item => item.reachable).length,
      answerChangedCount: items.filter(item => item.changed).length,
      targetLogProbabilityDelta: { minimum: deltas[0], p10: quantile(deltas, .1), median: quantile(deltas, .5), p90: quantile(deltas, .9), maximum: deltas.at(-1) },
      meanModifiedTargetProbability: items.reduce((sum, item) => sum + item.probability, 0) / items.length,
      jsDivergence: { median: quantile(jsd, .5), p90: quantile(jsd, .9), maximum: jsd.at(-1), exactlyZeroCount: items.filter(item => item.jsd === 0).length },
      maximumAbsoluteLogitChange: Math.max(...items.map(item => item.maxLogitChange)) };
  });
  const report = { format: 'transformer-microscope-head-coverage', schemaVersion: 1, recordedAt: new Date().toISOString(),
    modelId: manifest.modelId, modelRevision: manifest.modelRevision, split: 'test (held-out three-binding inputs)', inputs: inputs.length,
    intervention: 'head.scale factor 0 (ablation), one head at a time', predictionPosition: 'final query position', metric: 'natural-log target probability change',
    environment: { node: process.version, platform: `${platform()} ${release()}`, cpu: cpus()[0]?.model ?? 'unknown', backend: tf.getBackend(), tfjs: tf.version_core },
    mostHarmfulAllPositionHeadPerInput: Object.fromEntries([...strongest].sort((a, b) => b[1] - a[1])), configurations: summaries,
    limitations: ['One checkpoint, one controlled task, one prediction position, factor 0 only.', 'Single-head ablations do not measure interactions between heads.',
      'Effects measured on these inputs do not establish a semantic role for any head.'] };
  mkdirSync('.cache', { recursive: true });
  writeFileSync('.cache/head-coverage.json', JSON.stringify(report, null, 2) + '\n');
  console.info(JSON.stringify({ strongest: report.mostHarmfulAllPositionHeadPerInput, summaries: summaries.map(s => [s.label, s.positions, s.answerChangedCount, s.targetLogProbabilityDelta.median.toFixed(4), s.targetLogProbabilityDelta.minimum.toFixed(3), s.jsDivergence.exactlyZeroCount, s.reachableCount]) }));
});
