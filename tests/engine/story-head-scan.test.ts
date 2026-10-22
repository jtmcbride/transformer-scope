import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { scanStoryHeads, rankHeadScan, type HeadScanRow } from '../../src/story-head-scan';
import type { StorySelection } from '../../src/continuation';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-head-scan-reference.json';
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('all head trials match independent PyTorch, retain inherited heads, replace the same head, and release tensors', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  const engine = new RunEngine(loaded.artifact), memory = tf.memory().numTensors;
  try {
    for (const item of reference.cases) {
      const rows: HeadScanRow[] = [], source = item.source as StorySelection, snapshot = structuredClone(source);
      await scanStoryHeads(source, loaded.manifest.config, async (ids, edits) => {
        const base=await engine.runBaseline(ids); let modified: string | undefined;
        try { if(edits.length) modified=(await engine.compare({ baselineRunId:base.runId, interventions:edits })).modifiedRunId;
          return (await engine.inspectActivation(modified ?? base.runId,{stage:'logits'},{position:ids.length-1})).values;
        } finally { if(modified) engine.releaseRun(modified); engine.releaseRun(base.runId); }
      }, row => rows.push(row), () => false);
      expect(rows).toHaveLength(48); expect(source).toEqual(snapshot);
      rows.forEach((row,i) => { const expected=item.rows[i]; expect(row.edit).toEqual(expected.edit); expect(row.rank).toBe(expected.rank); expect(row.choice).toBe(expected.choice); expect(row.after).toBeCloseTo(expected.after,5); expect(row.before).toBeCloseTo(source.choice.probability,5); expect(row.delta).toBeCloseTo(expected.after-source.choice.probability,5); });
      const ranked=rankHeadScan(rows,'absolute'); expect(ranked.every((r,i) => !i || Math.abs(r.delta)<=Math.abs(ranked[i-1].delta))).toBe(true);
      expect(rankHeadScan(rows,'increase')[0].delta).toBe(Math.max(...rows.map(r=>r.delta)));
      expect(rankHeadScan(rows,'decrease')[0].delta).toBe(Math.min(...rows.map(r=>r.delta)));
      expect(tf.memory().numTensors).toBe(memory);
    }
  } finally { engine.dispose(); }
}, 60_000);
it('cancelled predictions cannot publish late rows and ranking ties are deterministic', async () => {
  const source=structuredClone(reference.cases[0].source) as StorySelection;
  const loaded=loadShippedArtifact('tinystories');
  const { chooseToken }=await import('../../src/continuation');
  const logits=[-99,-99,-99,1,2]; source.choice=chooseToken(logits,source.settings,0);
  let cancelled=false, calls=0; const rows: HeadScanRow[]=[];
  await scanStoryHeads(source, loaded.manifest.config, async () => { if(++calls===3) cancelled=true; return logits; },r=>rows.push(r),()=>cancelled);
  expect(rows).toHaveLength(1); expect(calls).toBe(3);
  const row=rows[0]; const ties=[{...row,edit:{...row.edit,head:2}}, {...row,edit:{...row.edit,head:1}}];
  expect(rankHeadScan(ties,'absolute').map(r=>r.edit.head)).toEqual([1,2]);
});
