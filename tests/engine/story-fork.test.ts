import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, selectionEdits, selectStoryToken, StoryContinuation } from '../../src/continuation';
import { firstForkDifference, forkEdits, selectForkToken, StoryFork, type ForkPlan } from '../../src/story-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-forks-reference.json';
import nested from '../fixtures/nested-story-forks-reference.json';

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

it('nested selections preserve exact prefixes, global sampling counters, all edits and one-prediction expiry', async () => {
  const config = loadShippedArtifact('tinystories').manifest.config;
  for (const example of nested.cases) {
    const parent = (example.parent.collection === 'root' ? reference.cases : nested.cases)[example.parent.case];
    const frame = { step: parent.original.choices.length, original: parent.original, edited: parent.edited };
    const selected = selectForkToken(parent.plan as ForkPlan, frame, example.parent.branch as 'edited', example.parent.index);
    expect(selected).toEqual(example.plan.source);
    const sourceSnapshot = structuredClone(selected);
    selected.edits![0].factor = 1.5;
    expect(selectForkToken(parent.plan as ForkPlan, frame, 'edited', example.parent.index)).toEqual(sourceSnapshot);
  }
  const once = nested.cases[1].plan as ForkPlan;
  expect(forkEdits(once, false, 0)).toHaveLength(1);
  expect(forkEdits(once, true, 0)).toHaveLength(2);
  expect(forkEdits(once, false, 1)).toEqual([]);
  expect(forkEdits(once, true, 1)).toEqual([]);
  const sustained = { ...once, duration: 'continuation' as const };
  expect(forkEdits(sustained, true, 1)).toEqual([once.edit]);
  expect(() => new StoryFork({ ...once, source: { ...once.source, continuationEdits: [{ ...once.edit, head: 99 }] } }, config)).toThrow('Head is outside');
  expect(() => selectForkToken(once, { step: 0, original: { tokenIds: once.source.prefix, choices: [] } }, 'original', 0)).toThrow('completed branch token');

  const settings = { temperature: .8, topP: .9, seed: 23, maxTokens: 5 };
  const logits = [-99, -99, -99, 0, .1, .2, .3];
  const first = new StoryFork({ ...once, source: { ...once.source, settings } }, config);
  while (!first.finished) await first.step(async () => logits);
  const token = selectForkToken(first.plan, first.frame, 'edited', 2);
  const next = new StoryFork({ ...once, source: token, maxTokens: 4 }, config);
  while (!next.finished) await next.step(async () => logits);
  expect(next.frame.original.choices).toEqual(first.frame.edited!.choices.slice(2));
  expect(next.frame.edited).toEqual(next.frame.original);
  expect(next.frame.original.choices[0]).toEqual(chooseToken(logits, settings, drawAt(settings.seed, token.step - 1)));
});

it('nested fork inspection and further changes match independent PyTorch with multiple inherited heads', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(nested.modelRevision);
  const engine = new RunEngine(loaded.artifact), before = tf.memory().numTensors;
  async function predict(ids: number[], edits: ReturnType<typeof selectionEdits>) {
    const base = await engine.runBaseline(ids);
    let modified: string | undefined;
    try {
      if (edits.length) modified = (await engine.compare({ baselineRunId: base.runId, interventions: edits })).modifiedRunId;
      return (await engine.inspectActivation(modified ?? base.runId, { stage: 'logits' }, { position: ids.length - 1 })).values;
    } finally { if (modified) engine.releaseRun(modified); engine.releaseRun(base.runId); }
  }
  try {
    for (const example of nested.cases) {
      const plan = example.plan as ForkPlan;
      const replay = chooseToken(await predict(plan.source.prefix, selectionEdits(plan.source)), plan.source.settings, drawAt(plan.source.settings.seed, plan.source.step - 1));
      expect(replay.tokenId).toBe(plan.source.choice.tokenId);
      expect(replay.probability).toBeCloseTo(plan.source.choice.probability, 5);
      const fork = new StoryFork(plan, loaded.manifest.config);
      while (!fork.finished) await fork.step(predict);
      for (const branch of ['original', 'edited'] as const) {
        expect(fork.frame[branch]!.tokenIds).toEqual(example[branch].tokenIds);
        fork.frame[branch]!.choices.forEach((choice, i) => expect(choice.probability).toBeCloseTo(example[branch].choices[i].probability, 5));
      }
      expect(tf.memory().numTensors).toBe(before);
    }
  } finally { engine.dispose(); }
}, 60_000);
