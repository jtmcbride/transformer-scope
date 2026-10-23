import { expect, it } from 'vitest';
import { chooseToken, drawAt, type StorySelection } from '../../src/continuation';
import { selectForkToken, type ForkPlan } from '../../src/story-fork';
import { compareInteractions, rankInteractionChanges, scanStoryHeatmaps, workspaceStoryPaths } from '../../src/story-heatmap-comparison';
import type { HeadPairScanRow } from '../../src/story-head-pair-scan';
import { headPairKey } from '../../src/head-pair-heatmap';
import { emptyWorkspace } from '../../src/story-workspace';
import { loadShippedArtifact } from './retrieval-artifact';
import comparison from '../fixtures/story-heatmap-comparison-reference.json';
import root from '../fixtures/continuation-reference.json';
import forks from '../fixtures/story-forks-reference.json';
const manifest = loadShippedArtifact('tinystories').manifest;
const asRows = (side: number) => comparison.cases[side].rows.map(row => ({ ...row,before:comparison.cases[side].source.choice.probability,delta:row.after-comparison.cases[side].source.choice.probability,rank:1,choice:3 })) as HeadPairScanRow[];
it('joins matching exact strengths and subtracts independent branched-token interactions in B-minus-A order', () => {
  const a=asRows(0),b=asRows(1),before=structuredClone([a,b]);
  const changes=compareInteractions(a,b);
  expect(changes).toHaveLength(120);
  for (const [i,row] of changes.entries()) {
    expect(row.key).toBe(headPairKey(comparison.differences[i].edits as HeadPairScanRow['edits']));
    expect(row.difference).toBe(comparison.differences[i].difference);
    const p0=comparison.cases.map(c => c.source.choice.probability);
    const interaction=(side:number) => {const r=comparison.cases[side].rows[i];return r.after-r.individualProbabilities[0]-r.individualProbabilities[1]+p0[side];};
    expect(row.difference).toBeCloseTo(interaction(1)-interaction(0),14);
  }
  for (const order of ['absolute','increase','decrease'] as const) {
    const score=(r:typeof changes[number]) => order === 'absolute' ? Math.abs(r.difference):order === 'increase' ? r.difference:-r.difference;
    expect(rankInteractionChanges([...changes].reverse(),order)[0].key).toBe([...changes].sort((x,y)=>score(y)-score(x))[0].key);
  }
  expect(compareInteractions(b,a).map(row=>row.difference)).toEqual(changes.map(row=>-row.difference));
  expect(compareInteractions(a,a).every(row=>row.difference === 0)).toBe(true);
  expect([a,b]).toEqual(before);
});
it('never treats a missing trial or a different strength assignment as a zero interaction', () => {
  const a=asRows(0),b=asRows(1);
  expect(compareInteractions(a,b.slice(0,3))).toHaveLength(3);
  expect(compareInteractions(a,[])).toEqual([]);
  const other=structuredClone(b[0]);other.edits[1].factor=2;
  expect(compareInteractions(a,[other])).toEqual([]);
  const ties=compareInteractions(a,b).slice(0,3).map(row=>({...row,difference:0}));
  expect(rankInteractionChanges([...ties].reverse(),'absolute').map(row=>row.key)).toEqual(ties.map(row=>row.key));
});
it('offers exact completed tokens from root and saved branch paths without changing source records', () => {
  const workspace=emptyWorkspace(manifest), parent=forks.cases[0];
  workspace.root={prompt:root.prompt,tokens:root.promptTokenIds,settings:root.settings,edit:root.edit as any,frame:{step:root.original.choices.length,original:root.original,edited:root.edited}};
  workspace.nodes=[{id:'saved',parentId:null,plan:parent.plan as ForkPlan,frame:{step:parent.original.choices.length,original:parent.original,edited:parent.edited}}];
  const before=structuredClone(workspace),paths=workspaceStoryPaths(workspace);
  expect(paths).toHaveLength(4);expect(paths[2].start).toBe(6);expect(paths[3].label).toBe('Comparison 1 · With head change');
  for (const [i,branch] of ['original','edited'].entries()) {
    const expected=selectForkToken(parent.plan as ForkPlan,workspace.nodes[0].frame,branch as 'original'|'edited',1);
    expect(paths[i+2].select(1)).toEqual(expected);
    expect(expected.prefix).toEqual(comparison.cases[i].source.prefix);
    expect(expected.edits).toEqual(comparison.cases[i].source.edits);
  }
  expect(workspace).toEqual(before);
  expect(workspaceStoryPaths(emptyWorkspace(manifest))).toEqual([]);
});
function synthetic() {
  const sources=comparison.cases.map(c=>structuredClone(c.source)) as [StorySelection,StorySelection];
  const logits=[-99,-99,-99,0,.1,.2];
  for(const source of sources)source.choice=chooseToken(logits,source.settings,drawAt(source.settings.seed,source.step-1));
  return {sources,logits};
}
it('scans each captured source in sequence with isolated predictors and releases each source scope', async () => {
  const {sources,logits}=synthetic(),snapshot=structuredClone(sources),rows:[HeadPairScanRow[],HeadPairScanRow[]]=[[],[]];let active=false,scopes=0;
  await scanStoryHeatmaps(sources,manifest.config,'mute',async(source,use)=>{
    expect(active).toBe(false);active=true;expect(source).toEqual(snapshot[scopes++]);
    // Caller mutation must not rewrite the second captured source.
    sources[1].prefix[0]=77;
    try {await use(async(ids)=>{expect(ids).toEqual(source.prefix);return logits;});}finally{active=false;}
  },(side,row)=>rows[side].push(row),()=>false);
  expect(scopes).toBe(2);expect(rows.map(side=>side.length)).toEqual([120,120]);expect(active).toBe(false);
});
it('retains only complete rows after cancellation or a second-source replay failure', async () => {
  const {sources,logits}=synthetic();let cancelled=false,scopes=0,calls=0;const rows:[HeadPairScanRow[],HeadPairScanRow[]]=[[],[]];
  await scanStoryHeatmaps(sources,manifest.config,'mute',async(_,use)=>{scopes++;await use(async()=>{if(++calls===4)cancelled=true;return logits;});},(side,row)=>rows[side].push(row),()=>cancelled);
  expect(scopes).toBe(1);expect(rows).toEqual([[],[]]);
  scopes=0; sources[1].choice.tokenId=3;
  await expect(scanStoryHeatmaps(sources,manifest.config,'mute',async(_,use)=>{scopes++;await use(async()=>logits);},(side,row)=>rows[side].push(row),()=>false)).rejects.toThrow('reproduce');
  expect(scopes).toBe(2);expect(rows[0]).toHaveLength(120);expect(rows[1]).toHaveLength(0);
});
