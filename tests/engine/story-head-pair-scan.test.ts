import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { headPairTrials, pairScanTotal, pairStrengthOptions, rankHeadPairScan, scanStoryHeadPairs, type PairStrengths, type HeadPairScanRow } from '../../src/story-head-pair-scan';
import { RunEngine } from '../../src/engine/runs';
import { StoryFork, type ForkPlan } from '../../src/story-fork';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-head-pair-scan-reference.json';
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('enumerates every distinct head pair with every requested strength assignment exactly once', () => {
  const config = loadShippedArtifact('tinystories').manifest.config;
  for (const option of pairStrengthOptions) {
    const pairs = [...headPairTrials(config, option.value)];
    expect(pairs).toHaveLength(pairScanTotal(config, option.value));
    expect(pairs).toHaveLength(120 * option.factors.length);
    expect(new Set(pairs.map(pair => JSON.stringify(pair))).size).toBe(pairs.length);
    for (const [a, b] of pairs) {
      expect(a.layer*config.heads+a.head).toBeLessThan(b.layer*config.heads+b.head);
      expect(option.factors.some(([x,y]) => a.factor === x && b.factor === y)).toBe(true);
    }
  }
  expect(pairScanTotal(config, 'all')).toBe(1080);
  expect(() => pairScanTotal(config, 'invalid' as PairStrengths)).toThrow('valid pair strengths');
});
it('all 720 joint and individual distributions and interactions match independent PyTorch without retained tensors', async () => {
  const loaded = loadShippedArtifact('tinystories'); expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  const engine = new RunEngine(loaded.artifact), memory = tf.memory().numTensors;
  try {
    for (const example of reference.cases) {
      const rows: HeadPairScanRow[] = [], source = structuredClone(example.source) as StorySelection;
      const base = await engine.runBaseline(source.prefix);
      const predict = async (ids: number[], edits: Parameters<typeof selectionEdits>[0]['edits'] = []) => {
        expect(ids).toEqual(source.prefix); let changed: string | undefined;
        try { if (edits.length) changed = (await engine.compare({ baselineRunId: base.runId, interventions: edits })).modifiedRunId;
          return (await engine.inspectActivation(changed ?? base.runId, { stage: 'logits' }, { position: ids.length-1 })).values;
        } finally { if (changed) engine.releaseRun(changed); }
      };
      try {
        // Replay checks compare CPU records to CPU. Cross-framework parity retains the existing tolerance.
        const replay = chooseToken(await predict(source.prefix, selectionEdits(source)), source.settings, drawAt(source.settings.seed, source.step-1));
        expect(replay.tokenId).toBe(example.source.choice.tokenId); expect(replay.probability).toBeCloseTo(example.source.choice.probability,5);
        source.choice = replay; const snapshot = structuredClone(source);
        await scanStoryHeadPairs(source, loaded.manifest.config, example.strengths as PairStrengths, predict, row => rows.push(row), () => false);
        expect(rows).toHaveLength(example.rows.length); expect(source).toEqual(snapshot);
        rows.forEach((row, i) => { const expected = example.rows[i];
          expect(row.edits).toEqual(expected.edits); expect(row.rank).toBe(expected.rank); expect(row.choice).toBe(expected.choice);
          expect(row.before).toBeCloseTo(example.source.choice.probability, 5); expect(row.after).toBeCloseTo(expected.after, 5); expect(row.delta).toBeCloseTo(expected.after-example.source.choice.probability, 5);
          row.individualProbabilities.forEach((p,j) => expect(p).toBeCloseTo(expected.individualProbabilities[j],5));
          const summed = row.individualProbabilities.reduce((sum,p) => sum + (p-row.before),0);
          expect(row.sumDelta).toBe(summed); expect(row.interaction).toBe(row.delta-summed);
          // I = pAB - pA - pB + p0: propagate the four independently bounded probability errors.
          const errorBudget = Math.abs(row.after-expected.after) + Math.abs(row.before-example.source.choice.probability)
            + row.individualProbabilities.reduce((sum,p,j) => sum + Math.abs(p-expected.individualProbabilities[j]),0);
          expect(Math.abs(row.interaction-expected.interaction)).toBeLessThanOrEqual(errorBudget+1e-12);
        });
      } finally { engine.releaseRun(base.runId); }
      for (const order of ['absolute', 'increase', 'decrease'] as const) {
        const expected = [...example.rows].sort((a,b) => order === 'absolute' ? Math.abs(b.after-source.choice.probability)-Math.abs(a.after-source.choice.probability) : order === 'increase' ? b.after-a.after : a.after-b.after);
        expect(rankHeadPairScan(rows, order)[0].edits).toEqual(expected[0].edits);
      }
      for (const order of ['interaction-absolute','interaction-positive','interaction-negative'] as const) {
        const score = (row: typeof example.rows[number]) => order === 'interaction-absolute' ? Math.abs(row.interaction) : order === 'interaction-positive' ? row.interaction : -row.interaction;
        const expected = [...example.rows].sort((a,b) => score(b)-score(a));
        expect(rankHeadPairScan(rows,order)[0].edits).toEqual(expected[0].edits);
      }
      const best = example.bestInteraction, fork = new StoryFork(best.plan as ForkPlan, loaded.manifest.config);
      while (!fork.finished) await fork.step(async (ids, edits) => {
        const base = await engine.runBaseline(ids); let changed: string | undefined;
        try {
          if (edits.length) changed = (await engine.compare({ baselineRunId: base.runId, interventions: edits })).modifiedRunId;
          return (await engine.inspectActivation(changed ?? base.runId, { stage: 'logits' }, { position: ids.length-1 })).values;
        } finally { if (changed) engine.releaseRun(changed); engine.releaseRun(base.runId); }
      });
      for (const branch of ['original','edited'] as const) {
        expect(fork.frame[branch]!.tokenIds).toEqual(best[branch].tokenIds);
        fork.frame[branch]!.choices.forEach((choice,i) => expect(choice.probability).toBeCloseTo(best[branch].choices[i].probability,5));
      }
      expect(tf.memory().numTensors).toBe(memory);
    }
  } finally { engine.dispose(); }
}, 120_000);
it('all-strength scanning measures joint trials separately, retains the sampling counter, and cancels without late rows', async () => {
  const config = loadShippedArtifact('tinystories').manifest.config;
  const source = structuredClone(reference.cases[0].source) as StorySelection;
  source.settings = { ...source.settings, temperature: .8, topP: 1, seed: 17 };
  const before = [-99,-99,-99,0,.1,.2]; source.choice = chooseToken(before, source.settings, drawAt(source.settings.seed, source.step-1));
  const rows: HeadPairScanRow[] = []; let calls = 0; const singles: string[] = [];
  await scanStoryHeadPairs(source, config, 'all', async (ids, edits) => {
    expect(ids).toEqual(source.prefix); ++calls;
    if (edits.length === 1) singles.push(`${edits[0].layer}:${edits[0].head}:${edits[0].factor}`);
    return edits.length ? [-99,-99,-99,0,edits.reduce((sum, edit) => sum + edit.factor/10, 0),.2] : before;
  }, row => rows.push(row), () => false);
  expect(rows).toHaveLength(1080); expect(calls).toBe(1129);
  expect(singles).toHaveLength(48); expect(new Set(singles).size).toBe(48);
  expect(new Set(rows.map(row => JSON.stringify(row.edits))).size).toBe(1080);
  let cancelled = false; calls = 0; rows.length = 0;
  await scanStoryHeadPairs(source, config, 'mute', async () => { if (++calls === 5) cancelled = true; return before; }, row => rows.push(row), () => cancelled);
  expect(calls).toBe(5); expect(rows).toHaveLength(1);
  for (const stopAt of [3,4]) {
    calls = 0; cancelled = false; rows.length = 0;
    await scanStoryHeadPairs(source, config, 'mute', async () => { if (++calls === stopAt) cancelled = true; return before; }, row => rows.push(row), () => cancelled);
    expect(calls).toBe(stopAt); expect(rows).toHaveLength(0);
  }
  calls = 0; rows.length = 0;
  await expect(scanStoryHeadPairs(source, config, 'mute', async () => { if (++calls === 4) throw new Error('individual failed'); return before; }, row => rows.push(row), () => false)).rejects.toThrow('individual failed');
  expect(rows).toHaveLength(0);
  calls = 0;
  await scanStoryHeadPairs(source, config, 'mute', async () => { ++calls; return before; }, () => { throw new Error('late row'); }, () => true);
  expect(calls).toBe(0);
});
it('rejects a stale source before measuring pairs and orders rounded or exact ties deterministically', async () => {
  const config = loadShippedArtifact('tinystories').manifest.config, source = structuredClone(reference.cases[0].source) as StorySelection;
  let calls = 0; source.choice.tokenId = 3;
  await expect(scanStoryHeadPairs(source, config, 'mute', async () => { ++calls; return [-99,-99,-99,0]; }, () => { throw new Error('unexpected row'); }, () => false)).rejects.toThrow('reproduce');
  expect(calls).toBe(1);
  const trials = [...headPairTrials(config, 'all')].slice(0, 18);
  const rows: HeadPairScanRow[] = trials.map(edits => ({ edits, before: .2, after: .2, delta: 0, rank: 1, choice: 3, individualProbabilities: [.2,.2], sumDelta: 0, interaction: 0 }));
  const ordered = rankHeadPairScan([...rows].reverse(), 'absolute');
  expect(ordered.map(row => row.edits)).toEqual(trials);
  expect(rows[0].edits).toEqual(trials[0]);
  for (const order of ['interaction-absolute','interaction-positive','interaction-negative'] as const) expect(rankHeadPairScan([...rows].reverse(),order).map(row => row.edits)).toEqual(trials);
  rows[0].interaction = -.3; rows[0].delta = .01; rows[1].interaction = .3; rows[1].delta = -.1;
  expect(rankHeadPairScan(rows,'interaction-absolute')[0]).toBe(rows[0]);
  expect(rankHeadPairScan(rows,'interaction-positive')[0]).toBe(rows[1]);
  expect(rankHeadPairScan(rows,'interaction-negative')[0]).toBe(rows[0]);
  expect(rankHeadPairScan(rows,'absolute')[0]).toBe(rows[1]);
});
