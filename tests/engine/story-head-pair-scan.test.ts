import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { headPairTrials, pairScanTotal, pairStrengthOptions, rankHeadPairScan, scanStoryHeadPairs, type PairStrengths, type HeadPairScanRow } from '../../src/story-head-pair-scan';
import { RunEngine } from '../../src/engine/runs';
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
it('all 720 joint distributions match independent PyTorch for unedited, inherited and mixed-strength scans with no retained tensors', async () => {
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
        });
      } finally { engine.releaseRun(base.runId); }
      for (const order of ['absolute', 'increase', 'decrease'] as const) {
        const expected = [...example.rows].sort((a,b) => order === 'absolute' ? Math.abs(b.after-source.choice.probability)-Math.abs(a.after-source.choice.probability) : order === 'increase' ? b.after-a.after : a.after-b.after);
        expect(rankHeadPairScan(rows, order)[0].edits).toEqual(expected[0].edits);
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
  const rows: HeadPairScanRow[] = []; let calls = 0;
  await scanStoryHeadPairs(source, config, 'all', async (ids, edits) => {
    expect(ids).toEqual(source.prefix); ++calls;
    return edits.length ? [-99,-99,-99,0,edits.reduce((sum, edit) => sum + edit.factor/10, 0),.2] : before;
  }, row => rows.push(row), () => false);
  expect(rows).toHaveLength(1080); expect(calls).toBe(1081);
  expect(new Set(rows.map(row => JSON.stringify(row.edits))).size).toBe(1080);
  let cancelled = false; calls = 0; rows.length = 0;
  await scanStoryHeadPairs(source, config, 'mute', async () => { if (++calls === 3) cancelled = true; return before; }, row => rows.push(row), () => cancelled);
  expect(calls).toBe(3); expect(rows).toHaveLength(1);
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
  const rows = trials.map(edits => ({ edits, before: .2, after: .2, delta: 0, rank: 1, choice: 3 }));
  const ordered = rankHeadPairScan([...rows].reverse(), 'absolute');
  expect(ordered.map(row => row.edits)).toEqual(trials);
  expect(rows[0].edits).toEqual(trials[0]);
});
