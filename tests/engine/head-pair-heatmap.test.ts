import { expect, it } from 'vitest';
import { headPairHeatmap, headPairKey, interactionFill } from '../../src/head-pair-heatmap';
import { headPairTrials, type HeadPairScanRow } from '../../src/story-head-pair-scan';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-head-pair-scan-reference.json';
const config = loadShippedArtifact('tinystories').manifest.config;
it('maps every independent trial to its exact two axes and strengths, preserving mixed assignments', () => {
  for (const example of reference.cases) {
    const rows = example.rows.map(row => ({ ...row, before: example.source.choice.probability, delta: row.after-example.source.choice.probability })) as HeadPairScanRow[];
    const factors = example.strengths === 'mute' ? [[0,0]] : [[0,2],[2,0]];
    const snapshot = structuredClone(rows);
    for (const [a,b] of factors) {
      const map = headPairHeatmap(rows,config,a,b);
      expect(map.measured).toBe(rows.length); expect(map.total).toBe(rows.length);
      for (const row of rows) {
        const first = row.edits[0], second = row.edits[1], i = first.layer*config.heads+first.head, j = second.layer*config.heads+second.head;
        const result = first.factor === a ? map.cells[i][j] : map.cells[j][i];
        expect(result).toBe(row);
        if (a === b) expect(map.cells[j][i]).toBe(row);
      }
      expect(map.cells.every((row,i) => row[i] === undefined)).toBe(true);
      expect(map.limit).toBe(Math.max(...rows.map(row => Math.abs(row.interaction))));
    }
    expect(rows).toEqual(snapshot);
  }
});
it('distinguishes pending cells from measured zeroes and shares the scale across all strength views', () => {
  const first = reference.cases[2].rows[0];
  const zero = { ...first, before: .2, delta: 0, interaction: 0 } as HeadPairScanRow;
  const rows = [zero, { ...zero, edits: [{ ...zero.edits[0], factor: 2 }, { ...zero.edits[1], factor: 0 }], interaction: -.25 } as HeadPairScanRow];
  const map = headPairHeatmap([zero],config,0,2);
  expect(map.cells[0][1]?.interaction).toBe(0); expect(map.cells[1][0]).toBeUndefined();
  expect(map.measured).toBe(1); expect(map.total).toBe(240);
  const forward = headPairHeatmap(rows,config,0,2), reverse = headPairHeatmap(rows,config,2,0);
  expect(forward.limit).toBe(.25); expect(reverse.limit).toBe(.25);
  expect(headPairKey(forward.cells[0][1]!.edits)).toBe(headPairKey(reverse.cells[1][0]!.edits));
  expect(headPairHeatmap([],config,1,1).measured).toBe(0);
  expect(interactionFill(0,0)).toBe(interactionFill(0,.25));
});
it('all nine axis-strength views retain all 1080 distinct all-strength trials without aggregation', () => {
  const rows = [...headPairTrials(config,'all')].map((edits,i) => ({ ...reference.cases[0].rows[0], edits, before: .2, delta: .1, interaction: i/2000 })) as HeadPairScanRow[];
  const shown = new Set<string>();
  for (const a of [0,1,2]) for (const b of [0,1,2]) {
    const map = headPairHeatmap(rows,config,a,b);
    expect(map.measured).toBe(a === b ? 120 : 240); expect(map.limit).toBe(1079/2000);
    for (const row of map.cells) for (const cell of row) if (cell) shown.add(headPairKey(cell.edits));
  }
  expect(shown.size).toBe(1080);
  expect([...shown].sort()).toEqual(rows.map(row => headPairKey(row.edits)).sort());
});
