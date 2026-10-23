import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { scanStoryHeadPairs, rankHeadPairScan, type HeadPairScanRow } from '../../src/story-head-pair-scan';
import { measureStoryProbabilities } from '../../src/story-probabilities';
import { StoryFork, type ForkPlan } from '../../src/story-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/chosen-token-pair-scan-reference.json';
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('ranks 240 chosen-target pair effects and branches with the actual sampler, matching independent PyTorch', async () => {
  const memory = tf.memory().numTensors, loaded = loadShippedArtifact('tinystories'), engine = new RunEngine(loaded.artifact);
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  try {
    for (const example of reference.cases) {
      const source = structuredClone(example.source) as StorySelection;
      expect(source.choice.tokenId).not.toBe(example.targetTokenId);
      const base = await engine.runBaseline(source.prefix);
      const predict = async (ids: number[], edits: ReturnType<typeof selectionEdits>) => {
        const same = JSON.stringify(ids) === JSON.stringify(source.prefix);
        const baseline = same ? base : await engine.runBaseline(ids); let changed: string | undefined;
        try {
          if (edits.length) changed = (await engine.compare({ baselineRunId: baseline.runId, interventions: edits })).modifiedRunId;
          return (await engine.inspectActivation(changed ?? baseline.runId, { stage: 'logits' }, { position: ids.length-1 })).values;
        } finally { if (changed) engine.releaseRun(changed); if (!same) engine.releaseRun(baseline.runId); }
      };
      try {
        source.choice = chooseToken(await predict(source.prefix, selectionEdits(source)), source.settings, drawAt(source.settings.seed,source.step-1));
        const snapshot = structuredClone(source), rows: HeadPairScanRow[] = [];
        await scanStoryHeadPairs(source, loaded.manifest.config, 'mute', predict, row => rows.push(row), () => false, example.targetTokenId);
        expect(rows).toHaveLength(120);
        rows.forEach((row,i) => {
          const expected = example.rows[i]; expect(row.edits).toEqual(expected.edits); expect(row.rank).toBe(expected.rank); expect(row.choice).toBe(expected.choice);
          for (const name of ['before','after','delta','sumDelta','interaction'] as const) expect(row[name]).toBeCloseTo(expected[name],5);
          row.individualProbabilities.forEach((value,j) => expect(value).toBeCloseTo(expected.individualProbabilities[j],5));
        });
        for (const order of ['increase','decrease'] as const) {
          const best = rankHeadPairScan(rows,order)[0], expected = example[order]; expect(best.edits).toEqual([expected.plan.edit,...expected.plan.additionalEdits]);
          const plan = { ...structuredClone(expected.plan), source } as ForkPlan;
          const report = await measureStoryProbabilities(plan,predict,() => false,example.targetTokenId);
          expect(report!.selected.tokenId).toBe(example.targetTokenId); expect(report!.selected.before).toBeCloseTo(best.before,10); expect(report!.selected.after).toBeCloseTo(best.after,10);
          expect(report!.beforeChoice).toBe(source.choice.tokenId); expect(report!.afterChoice).toBe(best.choice);
          const session = new StoryFork(plan,loaded.manifest.config); while (!session.finished) await session.step(predict);
          for (const branch of ['original','edited'] as const) {
            expect(session.frame[branch]!.tokenIds).toEqual(expected[branch].tokenIds);
            session.frame[branch]!.choices.forEach((choice,i) => expect(choice.probability).toBeCloseTo(expected[branch].choices[i].probability,5));
          }
        }
        expect(source).toEqual(snapshot);
        const stale = { ...source, choice: { ...source.choice, probability: source.choice.probability+.01 } };
        await expect(scanStoryHeadPairs(stale,loaded.manifest.config,'mute',predict,() => { throw Error('Stale result emitted'); },() => false,example.targetTokenId)).rejects.toThrow('reproduce');
        await expect(measureStoryProbabilities({ ...example.increase.plan, source:stale } as ForkPlan,predict,() => false,example.targetTokenId)).rejects.toThrow('reproduced');
      } finally { engine.releaseRun(base.runId); }
    }
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(memory);
},120000);
it('validates targets before inference and never emits a cancelled chosen-target trial', async () => {
  const config = loadShippedArtifact('tinystories').manifest.config, source = reference.cases[0].source as StorySelection;
  let calls = 0, emitted = 0;
  const predict = async () => { ++calls; return []; };
  for (const id of [0,2,-1,.5,NaN,config.vocabularySize]) await expect(scanStoryHeadPairs(source,config,'mute',predict,() => emitted++,() => false,id)).rejects.toThrow('valid output token');
  await scanStoryHeadPairs(source,config,'mute',predict,() => emitted++,() => true,471);
  expect(calls).toBe(0); expect(emitted).toBe(0);
});
