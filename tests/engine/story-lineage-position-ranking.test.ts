import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { lineagePositionBranchDraft, lineagePositionChanges, lineagePositionScore, rankLineagePositions } from '../../src/story-lineage-position-ranking';
import { captureLineageStories, originalSelection, type LineageProbabilityRows, type LineageSnapshot } from '../../src/story-lineage-probabilities';
import { parentSelection } from '../../src/story-parent-probabilities';
import { selectForkToken } from '../../src/story-fork';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { SharedStoryFork } from '../../src/story-shared-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import { lineageProbabilityWorkspace } from '../support/lineage-probability-workspace';
import traces from '../fixtures/lineage-probabilities-reference.json';
import reference from '../fixtures/lineage-position-rebranches-reference.json';

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
const orders=['absolute','increase','decrease'] as const;
function measured(example:typeof traces.cases[number],snapshot:LineageSnapshot):LineageProbabilityRows{
  return example.rows.map((track,side)=>Object.fromEntries((['original','parent','child'] as const).map(role=>[role,track[role].map(point=>({offset:point.offset,targetTokenId:example.targetTokenId,probability:point.probability,
    source:role==='original'?originalSelection(snapshot,side,point.offset):role==='parent'?parentSelection(snapshot,side,point.offset):selectForkToken(snapshot.children[side].plan,snapshot.children[side].frame,example.branch as StorySelection['branch'],point.offset)}))]))) as LineageProbabilityRows;
}
it('ranks four original/latest values without requiring parents, preserving zeroes, raw signed order, offset ties and missing peers',()=>{
  const track=(deltas:[number,number][],step=3)=>({parent:[],original:deltas.map(([offset])=>({offset,targetTokenId:471,probability:.5,source:{step:step+offset} as StorySelection})),child:deltas.map(([offset,delta])=>({offset,targetTokenId:471,probability:.5+delta,source:{step:step+offset} as StorySelection}))});
  const rows:LineageProbabilityRows=[track([[4,2e-10],[1,-.25],[0,.125],[2,0],[3,1e-10],[5,.1],[6,.2],[7,.3],[8,.1],[9,.1],[-1,.3]]),track([[0,-.25],[2,0],[1,.125],[4,0],[3,0],[5,.1],[6,.2],[7,.3],[8,.1],[-1,.3]],100)];
  rows[1].child.find(p=>p.offset===5)!.targetTokenId=1;rows[1].child.find(p=>p.offset===6)!.source.step++;rows[1].child.find(p=>p.offset===7)!.probability=NaN;rows[1].original.find(p=>p.offset===8)!.probability=-.1;
  const before=structuredClone(rows),positions=lineagePositionChanges(rows,471);
  expect(positions.map(p=>p.offset)).toEqual([0,1,2,3,4]);expect(positions[2].a.difference).toBe(0);
  expect(rankLineagePositions(positions,'absolute').map(p=>p.offset)).toEqual([0,1,4,3,2]);expect(rankLineagePositions(positions,'increase').map(p=>p.offset)).toEqual([0,1,4,3,2]);expect(rankLineagePositions(positions,'decrease').map(p=>p.offset)).toEqual([0,1,2,3,4]);
  expect(lineagePositionChanges([rows[0],{original:[],parent:[],child:[]}],471)).toEqual([]);expect(rows).toEqual(before);
});
it('ranks all 56 independent original/latest traces, including absent parents, asymmetric originals, EOS and unchanged references',()=>{
  let beyondParent=false,missingOriginal=false;
  for(const example of traces.cases){
    const workspace=lineageProbabilityWorkspace(example.rebranchReferenceIndex,example.targetTokenId),snapshot=captureLineageStories(workspace,workspace.sharedForks!.at(-1)!);
    const rows=measured(example,snapshot),positions=lineagePositionChanges(rows,example.targetTokenId),count=Math.min(...example.rows.flatMap(side=>[side.original.length,side.child.length]));
    expect(positions).toHaveLength(count);positions.forEach(p=>{expect(p.a.difference).toBe(example.rows[0].child[p.offset].probability-example.rows[0].original[p.offset].probability);expect(p.b.difference).toBe(example.rows[1].child[p.offset].probability-example.rows[1].original[p.offset].probability);});
    beyondParent ||=positions.some(p=>example.rows.some(track=>p.offset>=track.parent.length));missingOriginal ||=example.rows.some(track=>track.original.length<track.child.length);
    for(const order of orders){const ranked=rankLineagePositions(positions,order);for(let i=1;i<ranked.length;i++)expect((lineagePositionScore(ranked[i-1],order)-lineagePositionScore(ranked[i],order))*(order==='decrease'?-1:1)).toBeGreaterThanOrEqual(0);}
  }
  expect(beyondParent).toBe(true);expect(missingOriginal).toBe(true);
});
it('rebranches the ranked latest paths against 26 independent PyTorch four-path rollouts without changing original or parent stories',async()=>{
  const memory=tf.memory().numTensors,loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),cache=new Map<string,ArrayLike<number>>();
  const predict=async(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions'])=>{
    const key=JSON.stringify([ids,edits]);if(cache.has(key))return cache.get(key)!;const base=await engine.runBaseline(ids);let changed:string|undefined;
    try{if(edits.length)changed=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;const values=(await engine.inspectActivation(changed??base.runId,{stage:'logits'},{position:ids.length-1})).values;cache.set(key,values);return values;}
    finally{if(changed)engine.releaseRun(changed);engine.releaseRun(base.runId);}
  };
  try{
    expect(reference.modelRevision).toBe(loaded.manifest.modelRevision);expect(reference.cases).toHaveLength(26);let beyondParent=false;
    for(const example of reference.cases){
      const workspace=lineageProbabilityWorkspace(example.rebranchReferenceIndex,example.targetTokenId),snapshot=captureLineageStories(workspace,workspace.sharedForks!.at(-1)!);
      const trace=traces.cases.find(row=>row.rebranchReferenceIndex===example.rebranchReferenceIndex&&row.branch===example.branch&&row.targetTokenId===example.targetTokenId)!;
      const positions=lineagePositionChanges(measured(trace,snapshot),example.targetTokenId),ranked=rankLineagePositions(positions,example.order as typeof orders[number]);
      expect(ranked.map(p=>p.offset)).toEqual(example.ranked.map(p=>p.offset));ranked.forEach((p,i)=>expect([p.a.difference,p.b.difference]).toEqual([example.ranked[i].a,example.ranked[i].b]));
      beyondParent ||=trace.rows.some(track=>example.offset>=track.parent.length);
      const before=structuredClone(snapshot),draft=lineagePositionBranchDraft(snapshot,example.branch as StorySelection['branch'],ranked[0]);
      expect(draft.sources).toEqual(example.plans.map(p=>p.source));expect(draft.parents).toEqual(snapshot.children.map(n=>n.id));expect(draft.origin).toEqual({kind:'lineage-probability',groupId:snapshot.group.id,branch:example.branch,index:example.offset});
      for(const source of draft.sources){const actual=chooseToken(await predict(source.prefix,selectionEdits(source)),source.settings,drawAt(source.settings.seed,source.step-1));expect(actual.tokenId).toBe(source.choice.tokenId);expect(actual.probability).toBeCloseTo(source.choice.probability,5);source.choice=actual;}
      const pair=draft.edits.map(edit=>({...edit,factor:example.strengths==='mute'?0:edit.factor}));
      const plans=draft.sources.map(source=>({source,edit:pair[0],additionalEdits:[pair[1]],duration:example.duration,maxTokens:3})) as ConstructorParameters<typeof SharedStoryFork>[0],session=new SharedStoryFork(plans,loaded.manifest.config);
      while(!session.finished)await session.step(predict);
      session.frames.forEach((frame,side)=>{for(const branch of ['original','edited'] as const){expect(frame[branch]!.tokenIds).toEqual(example.frames[side][branch].tokenIds);frame[branch]!.choices.forEach((choice,i)=>expect(choice.probability).toBeCloseTo(example.frames[side][branch].choices[i].probability,5));}});
      expect(snapshot).toEqual(before);
    }
    expect(beyondParent).toBe(true);
  }finally{engine.dispose();}expect(tf.memory().numTensors).toBe(memory);
},120000);
it('rejects stale original/latest prefixes, choices, targets, expiry, draws, scores and pair identities before setting up a branch',()=>{
  const workspace=lineageProbabilityWorkspace(),snapshot=captureLineageStories(workspace,workspace.sharedForks!.at(-1)!);
  const example=traces.cases.find(row=>row.rebranchReferenceIndex===2&&row.branch==='edited'&&row.targetTokenId===471)!;
  const point=rankLineagePositions(lineagePositionChanges(measured(example,snapshot),471),'absolute')[0];expect(point.offset).toBe(2);
  for(const mutate of [
    (p:typeof point)=>{p.a.comparison.source.prefix[0]=33;},(p:typeof point)=>{p.b.comparison.source.choice.probability*=.5;},
    (p:typeof point)=>{p.b.comparison.targetTokenId=1;},(p:typeof point)=>{p.a.original.source.prefix[0]=33;},
    (p:typeof point)=>{p.b.comparison.source.step++;},(p:typeof point)=>{p.a.comparison.source.settings.seed++;},
    (p:typeof point)=>{p.a.comparison.source.continuationEdits=[];},(p:typeof point)=>{p.a.difference=0;},
    (p:typeof point)=>{p.b.original.targetTokenId=1;},(p:typeof point)=>{p.b.original.probability=NaN;},
    (p:typeof point)=>{p.b.original.source.choice.probability*=.5;},
  ]){const changed=structuredClone(point);mutate(changed);expect(()=>lineagePositionBranchDraft(snapshot,'edited',changed)).toThrow('no longer matches');}
  const bad=structuredClone(snapshot);bad.group.nodeIds[0]='another-comparison';expect(()=>lineagePositionBranchDraft(bad,'edited',point)).toThrow('another comparison');
  const pair=structuredClone(snapshot);pair.children[1].plan.edit.factor=0;expect(()=>lineagePositionBranchDraft(pair,'edited',point)).toThrow('same selected pair');
  expect(()=>lineagePositionBranchDraft(snapshot,'original',point)).toThrow('no longer matches');
  const original=structuredClone(snapshot),draft=lineagePositionBranchDraft(snapshot,'edited',point);draft.sources[0].prefix[0]=33;draft.edits[0].factor=0;expect(snapshot).toEqual(original);
});
