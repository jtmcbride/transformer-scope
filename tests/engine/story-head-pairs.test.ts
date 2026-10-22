import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, selectionEdits } from '../../src/continuation';
import { forkEdits, proposedEdits, selectForkToken, StoryFork, type ForkPlan } from '../../src/story-fork';
import { measureStoryProbabilities } from '../../src/story-probabilities';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-head-pairs-reference.json';
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('two edits replace their own heads, retain other settings, and expire together without mutating history', () => {
  const config = loadShippedArtifact('tinystories').manifest.config;
  const plan = structuredClone(reference.cases[2].plan) as ForkPlan;
  const source = structuredClone(plan.source);
  const fork = new StoryFork(plan, config);
  expect(forkEdits(plan, true, 0)).toEqual([plan.source.edits![1], plan.edit, plan.additionalEdits![0]]);
  plan.edit.factor = 2; plan.additionalEdits![0].factor = 2;
  expect(fork.plan.edit.factor).toBe(1); expect(fork.plan.additionalEdits![0].factor).toBe(0);
  expect(fork.plan.source).toEqual(source);
  const once = { ...fork.plan, duration: 'prediction' as const };
  expect(forkEdits(once, true, 1)).toEqual(selectionEdits(source, true));
  const expiry = reference.cases[3].plan as ForkPlan;
  expect(forkEdits(expiry, true, 0)).toHaveLength(3);
  expect(forkEdits(expiry, true, 1)).toEqual(proposedEdits(expiry));
  expect(forkEdits(expiry, false, 1)).toEqual([]);
});
it('rejects overlapping, oversized, malformed, or invalid proposed pairs before any prediction', () => {
  const config = loadShippedArtifact('tinystories').manifest.config;
  const plan = reference.cases[0].plan as ForkPlan;
  expect(() => new StoryFork({ ...plan, additionalEdits: [{ ...plan.edit, factor: 2 }] }, config)).toThrow('different heads');
  expect(() => new StoryFork({ ...plan, additionalEdits: [plan.edit, plan.edit] }, config)).toThrow('at most two');
  expect(() => new StoryFork({ ...plan, additionalEdits: null as any }, config)).toThrow('at most two');
  expect(() => new StoryFork({ ...plan, additionalEdits: [{ ...plan.additionalEdits![0], factor: NaN }] }, config)).toThrow('strength');
  expect(() => new StoryFork({ ...plan, additionalEdits: [{ ...plan.additionalEdits![0], tokens: { kind: 'indices', indices: [0] } }] }, config)).toThrow('all-position');
});
it('joint distributions, alternatives and both-duration rollouts match independent PyTorch without retained tensors', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  const engine = new RunEngine(loaded.artifact), memory = tf.memory().numTensors;
  async function predict(ids: number[], edits: Parameters<RunEngine['compare']>[0]['interventions']) {
    const base = await engine.runBaseline(ids); let modified: string | undefined;
    try { if (edits.length) modified = (await engine.compare({ baselineRunId: base.runId, interventions: edits })).modifiedRunId;
      return (await engine.inspectActivation(modified ?? base.runId, { stage: 'logits' }, { position: ids.length - 1 })).values;
    } finally { if (modified) engine.releaseRun(modified); engine.releaseRun(base.runId); }
  }
  try {
    for (const example of reference.cases) {
      const plan = example.plan as ForkPlan;
      const report = (await measureStoryProbabilities(plan, predict))!;
      expect(report.beforeChoice).toBe(example.report.beforeChoice); expect(report.afterChoice).toBe(example.report.afterChoice);
      expect(report.rows.map(r => r.tokenId).sort((a,b) => a-b)).toEqual(example.report.rows.map(r => r.tokenId).sort((a,b) => a-b));
      for (const expected of example.report.rows) {
        const actual = report.rows.find(r => r.tokenId === expected.tokenId)!;
        expect(actual.beforeRank).toBe(expected.beforeRank); expect(actual.afterRank).toBe(expected.afterRank);
        for (const key of ['before', 'after', 'delta'] as const) expect(actual[key]).toBeCloseTo(expected[key], 5);
      }
      report.individuals.forEach((row, i) => { expect(row.edit).toEqual(example.report.individuals[i].edit); expect(row.probability).toBeCloseTo(example.report.individuals[i].probability, 5); expect(row.delta).toBeCloseTo(example.report.individuals[i].delta, 5); });
      expect(report.interaction).toBeCloseTo(example.report.interaction, 5);
      const fork = new StoryFork(plan, loaded.manifest.config);
      while (!fork.finished) await fork.step(predict);
      for (const branch of ['original', 'edited'] as const) {
        expect(fork.frame[branch]!.tokenIds).toEqual(example[branch].tokenIds);
        fork.frame[branch]!.choices.forEach((choice,i) => expect(choice.probability).toBeCloseTo(example[branch].choices[i].probability,5));
      }
      const selected = selectForkToken(fork.plan, fork.frame, 'edited', 0);
      expect(selected.edits).toEqual(forkEdits(plan, true, 0));
      expect(selected.continuationEdits).toEqual(forkEdits(plan, true, 1));
      const restored = new StoryFork(plan, loaded.manifest.config); restored.restore(fork.frame);
      expect(restored.frame).toEqual(fork.frame);
      expect(tf.memory().numTensors).toBe(memory);
    }
  } finally { engine.dispose(); }
}, 60_000);
it('two-head seeded continuations share global draws and commit neither path when the pair computation fails', async () => {
  const config = loadShippedArtifact('tinystories').manifest.config;
  const plan = structuredClone(reference.cases[0].plan) as ForkPlan;
  plan.edit.factor = 1; plan.additionalEdits![0].factor = 1;
  plan.source.settings = { temperature: .8, topP: .9, seed: 23, maxTokens: 6 };
  const fork = new StoryFork(plan, config), snapshot = fork.frame;
  await expect(fork.step(async (_, edits) => { if (edits.length === 2) throw new Error('pair failed'); return [-99,-99,-99,0,.1,.2]; })).rejects.toThrow('pair failed');
  expect(fork.frame).toEqual(snapshot);
  while (!fork.finished) await fork.step(async () => [-99,-99,-99,0,.1,.2]);
  expect(fork.frame.edited).toEqual(fork.frame.original);
});
it('cancelled probability measurements publish no partial pair, and replay mismatch stops further work', async () => {
  const plan = structuredClone(reference.cases[0].plan) as ForkPlan;
  const logits = [-99,-99,-99,0,.1,.2]; plan.source.choice = chooseToken(logits, plan.source.settings, 0);
  let calls = 0;
  expect(await measureStoryProbabilities(plan, async () => { ++calls; return logits; }, () => calls >= 3)).toBeNull();
  expect(calls).toBe(3);
  plan.source.choice.probability = 0;
  await expect(measureStoryProbabilities(plan, async () => { ++calls; return logits; })).rejects.toThrow('reproduced');
  expect(calls).toBe(4);
});
