import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { branchProbabilities, headPairInteraction } from '../../src/branch-probabilities';
import { forkEdits, type ForkPlan } from '../../src/story-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/branch-probabilities-reference.json';
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('measures non-additivity in raw probability rather than confusing its sign with the joint change', () => {
  expect(headPairInteraction(.5, .6, [.55, .55]).interaction).toBeCloseTo(0, 14);
  // Positive interaction despite a net decrease; negative interaction despite a net increase.
  const positive = headPairInteraction(.5, .4, [.3, .3]);
  expect(positive.sumDelta).toBeCloseTo(-.4,14); expect(positive.interaction).toBeCloseTo(.3,14);
  expect(headPairInteraction(.5, .6, [.7, .7]).interaction).toBeCloseTo(-.3, 14);
  expect(headPairInteraction(.5, .51, [.7, .7]).interaction).toBeCloseTo(-.39, 14);
  for (const probabilities of [[], [.3], [.3, .4, .5], [.3, NaN], [.3, 1.1]]) {
    expect(() => headPairInteraction(.5, .6, probabilities)).toThrow('two individual probabilities');
  }
});
it('keeps raw softmax probabilities, excludes BOS/PAD from ranks, resolves ties, and reports sampled choices', () => {
  const settings = { temperature: .8, topP: 1, seed: 3, maxTokens: 3 };
  const before = [20, 1, 19, 1, 0, -.1, -.2, -.3, -.4, -.5];
  const after = [20, -.1, 19, 1, 2, 0, 0, 0, 0, 0];
  const r = branchProbabilities(before, after, 9, settings, 6);
  expect(r.selected.beforeRank).toBe(8); expect(r.selected.afterRank).toBe(7);
  expect(r.rows.some(row => row.tokenId === 0 || row.tokenId === 2)).toBe(false);
  expect(r.rows.find(row => row.tokenId === 1)!.beforeRank).toBe(1);
  expect(r.rows.find(row => row.tokenId === 3)!.beforeRank).toBe(2);
  expect(r.selected.before).toBeLessThan(1e-8);
  expect(r.rows.some(row => row.tokenId === r.beforeChoice)).toBe(true);
  const identity = branchProbabilities(before, before, 9, settings, 6);
  expect(identity.beforeChoice).toBe(identity.afterChoice);
  expect(identity.rows.every(row => row.delta === 0)).toBe(true);
});
it('selected token, alternatives, ranks and percentage changes match independent same-prefix PyTorch predictions', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  const engine = new RunEngine(loaded.artifact), before = tf.memory().numTensors;
  async function logits(plan: ForkPlan, changed: boolean) {
    const base = await engine.runBaseline(plan.source.prefix);
    let modified: string | undefined;
    try {
      const edits = forkEdits(plan, changed, 0);
      if (edits.length) modified = (await engine.compare({ baselineRunId: base.runId, interventions: edits })).modifiedRunId;
      return (await engine.inspectActivation(modified ?? base.runId, { stage: 'logits' }, { position: plan.source.prefix.length - 1 })).values;
    } finally { if (modified) engine.releaseRun(modified); engine.releaseRun(base.runId); }
  }
  try {
    for (const example of reference.cases) {
      const plan = example.plan as ForkPlan;
      const report = branchProbabilities(await logits(plan, false), await logits(plan, true), plan.source.choice.tokenId, plan.source.settings, plan.source.step);
      expect(report.beforeChoice).toBe(example.beforeChoice); expect(report.afterChoice).toBe(example.afterChoice);
      expect(report.rows.map(r => r.tokenId).sort((a,b) => a-b)).toEqual(example.rows.map(r => r.tokenId).sort((a,b) => a-b));
      for (const expected of example.rows) {
        const actual = report.rows.find(r => r.tokenId === expected.tokenId)!;
        expect(actual.beforeRank).toBe(expected.beforeRank); expect(actual.afterRank).toBe(expected.afterRank);
        for (const key of ['before', 'after', 'delta'] as const) expect(actual[key]).toBeCloseTo(expected[key], 5);
      }
      expect(tf.memory().numTensors).toBe(before);
    }
  } finally { engine.dispose(); }
}, 60_000);
