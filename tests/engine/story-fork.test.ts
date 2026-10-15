import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { selectStoryToken, StoryContinuation } from '../../src/continuation';
import { firstForkDifference, forkEdits, StoryFork, type ForkPlan } from '../../src/story-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-forks-reference.json';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('historical fork preserves sampling counter, exact prefixes and source history', async () => {
  const { manifest } = loadShippedArtifact('tinystories');
  const settings = { temperature: 1, topP: 1, seed: 42, maxTokens: 6 };
  const source = new StoryContinuation([0, 243, 162], settings, 64);
  const logits = [-99, -99, -99, 0, 0, 0, 0];
  while (!source.finished) await source.step(async () => logits);
  const before = source.frame;
  const plan: ForkPlan = { source: selectStoryToken(before, 3, 'original', 2, settings), edit: { id: 'identity', kind: 'head.scale', layer: 0, head: 0, tokens: { kind: 'all' }, factor: 1 }, duration: 'continuation', maxTokens: 4 };
  const fork = new StoryFork(plan, manifest.config);
  plan.source.prefix[1] = 99; plan.source.settings.seed = 77; plan.edit.factor = 0;
  while (!fork.finished) await fork.step(async () => logits);
  expect(fork.frame.original.tokenIds).toEqual(before.original.tokenIds);
  expect(fork.frame.edited).toEqual(fork.frame.original);
  expect(fork.frame.original.choices).toEqual(before.original.choices.slice(2));
  expect(firstForkDifference(fork.frame)).toBeNull();
  expect(source.frame).toEqual(before);
  expect(fork.plan.source.prefix.slice(0, 3)).toEqual([0, 243, 162]);
  expect(fork.plan.edit.factor).toBe(1);
});

it('fork edits replace the same head, retain other heads, expire after one prediction, and commit pairs atomically', async () => {
  const plan = structuredClone(reference.cases[2].plan) as ForkPlan;
  const inherited = plan.source.edit!;
  expect(forkEdits(plan, false, 0)).toEqual([inherited]);
  expect(forkEdits(plan, true, 0)).toEqual([inherited, plan.edit]);
  plan.duration = 'prediction';
  expect(forkEdits(plan, true, 1)).toEqual([inherited]);
  plan.edit.layer = inherited.layer; plan.edit.head = inherited.head; plan.edit.factor = 1;
  expect(forkEdits(plan, true, 0)).toEqual([plan.edit]);
  const config = loadShippedArtifact('tinystories').manifest.config;
  plan.maxTokens = 3;
  const fork = new StoryFork(plan, config);
  await expect(fork.step(async (_, edits) => { if (edits[0].factor === 1) throw new Error('cancelled'); return [-99, -99, -99, 1]; })).rejects.toThrow('cancelled');
  expect(fork.frame.step).toBe(0);
  await fork.step(async (_, edits) => edits[0].factor === 1 ? [-99, 1, -99, -99] : [-99, -99, -99, 1]);
  expect(fork.frame.edited?.finish).toBe('end of story');
  expect(firstForkDifference(fork.frame)).toBe(0);
  const seen: unknown[] = [];
  await fork.step(async (_, edits) => { seen.push(edits); return [-99, -99, -99, 1]; });
  expect(seen).toEqual([[inherited]]);
  expect(fork.frame.edited?.choices).toHaveLength(1);
  expect(() => new StoryFork({ ...plan, maxTokens: 0 }, config)).toThrow('between 1 and 64');
  expect(() => new StoryFork({ ...plan, edit: { ...plan.edit, factor: 9 } }, config)).toThrow('strength');
  const edge = new StoryFork({ ...plan, source: { ...plan.source, prefix: [0, ...Array(62).fill(3)] } }, config);
  await edge.step(async () => [-99, -99, -99, 1]);
  expect(edge.frame.original.finish).toBe('context limit'); expect(edge.finished).toBe(true);
});

it('both historical edit durations and inherited/replaced heads match independent PyTorch forks without retained tensors', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  const engine = new RunEngine(loaded.artifact), before = tf.memory().numTensors;
  try {
    for (const example of reference.cases) {
      const fork = new StoryFork(example.plan as ForkPlan, loaded.manifest.config);
      while (!fork.finished) await fork.step(async (tokenIds, edits) => {
        const baseline = await engine.runBaseline(tokenIds);
        let modified: string | undefined;
        try {
          if (edits.length) modified = (await engine.compare({ baselineRunId: baseline.runId, interventions: edits })).modifiedRunId;
          return (await engine.inspectActivation(modified ?? baseline.runId, { stage: 'logits' }, { position: tokenIds.length - 1 })).values;
        } finally { if (modified) engine.releaseRun(modified); engine.releaseRun(baseline.runId); }
      });
      for (const key of ['original', 'edited'] as const) {
        expect(fork.frame[key]!.tokenIds).toEqual(example[key].tokenIds);
        fork.frame[key]!.choices.forEach((choice, i) => expect(choice.probability).toBeCloseTo(example[key].choices[i].probability, 5));
      }
      expect(fork.frame.original.choices[0].tokenId).toBe(example.plan.source.choice.tokenId);
      expect(tf.memory().numTensors).toBe(before);
    }
  } finally { engine.dispose(); }
}, 60_000);
