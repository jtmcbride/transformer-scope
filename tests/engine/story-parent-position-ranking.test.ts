import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { parentPositionBranchDraft, parentPositionChanges, parentPositionScore, rankParentPositions } from '../../src/story-parent-position-ranking';
import { captureParentStories, parentSelection, type ParentProbabilityRows, type ParentSnapshot } from '../../src/story-parent-probabilities';
import { selectForkToken } from '../../src/story-fork';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { SharedStoryFork } from '../../src/story-shared-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import { parentProbabilityWorkspace } from '../support/parent-probability-workspace';
import traces from '../fixtures/parent-probabilities-reference.json';
import reference from '../fixtures/parent-position-rebranches-reference.json';

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
const orders=['absolute','increase','decrease'] as const;
function measured(example:typeof traces.cases[number],snapshot:ParentSnapshot):ParentProbabilityRows{
  return example.rows.map((track,side)=>Object.fromEntries((['parent','child'] as const).map(role=>[role,track[role].map(point=>({offset:point.offset,targetTokenId:example.targetTokenId,probability:point.probability,
    source:role==='parent'?parentSelection(snapshot,side,point.offset):selectForkToken(snapshot.children[side].plan,snapshot.children[side].frame,example.childBranch as 'original'|'edited',point.offset)}))]))) as ParentProbabilityRows;
}
it('joins only four measured values for one target and matching offsets, retains zeroes, and ranks signed raw effects with offset ties',()=>{
  const track=(deltas:[number,number][],step=3)=>({parent:deltas.map(([offset])=>({offset,targetTokenId:471,probability:.5,source:{step:step+offset} as StorySelection})),child:deltas.map(([offset,delta])=>({offset,targetTokenId:471,probability:.5+delta,source:{step:step+offset} as StorySelection}))});
  const rows:ParentProbabilityRows=[track([[4,2e-10],[1,-.25],[0,.125],[2,0],[3,1e-10],[5,.1],[6,.2],[7,.3],[8,.1],[9,.1]]),track([[0,-.25],[2,0],[1,.125],[4,0],[3,0],[5,.1],[6,.2],[7,.3],[8,.1]],100)];
  rows[1].child.find(p=>p.offset===5)!.targetTokenId=1;rows[1].child.find(p=>p.offset===6)!.source.step++;rows[1].child.find(p=>p.offset===7)!.probability=NaN;rows[1].child.find(p=>p.offset===8)!.probability=1.1;
  const before=structuredClone(rows),positions=parentPositionChanges(rows,471);
  expect(positions.map(p=>p.offset)).toEqual([0,1,2,3,4]);expect(positions[2].a.difference).toBe(0);
  expect(rankParentPositions(positions,'absolute').map(p=>p.offset)).toEqual([0,1,4,3,2]);
  expect(rankParentPositions(positions,'increase').map(p=>p.offset)).toEqual([0,1,4,3,2]);
  expect(rankParentPositions(positions,'decrease').map(p=>p.offset)).toEqual([0,1,2,3,4]);
  expect(parentPositionChanges([rows[0],{parent:[],child:[]}],471)).toEqual([]);expect(rows).toEqual(before);
});
it('ranks every independent parent/new trace, including unequal tails, root parents, EOS and zero-change reference paths',()=>{
  for(const example of traces.cases){
    const workspace=parentProbabilityWorkspace(example.referenceIndex,example.targetTokenId),snapshot=captureParentStories(workspace,workspace.sharedForks!.at(-1)!);
    const rows=measured(example,snapshot),positions=parentPositionChanges(rows,example.targetTokenId);
    expect(positions).toHaveLength(Math.min(...example.rows.flatMap(side=>[side.parent.length,side.child.length])));
    positions.forEach(p=>{expect(p.a.difference).toBe(example.rows[0].child[p.offset].probability-example.rows[0].parent[p.offset].probability);expect(p.b.difference).toBe(example.rows[1].child[p.offset].probability-example.rows[1].parent[p.offset].probability);});
    for(const order of orders){const ranked=rankParentPositions(positions,order);for(let i=1;i<ranked.length;i++)expect((parentPositionScore(ranked[i-1],order)-parentPositionScore(ranked[i],order))*(order==='decrease'?-1:1)).toBeGreaterThanOrEqual(0);}
    if(example.childBranch==='original')expect(rankParentPositions(positions,'absolute').map(p=>p.offset)).toEqual(positions.map(p=>p.offset));
  }
});
it('rebranches ranked new paths against 24 independent four-path PyTorch rollouts, preserving source settings and immutable ancestry',async()=>{
  const memory=tf.memory().numTensors,loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),cache=new Map<string,ArrayLike<number>>();
  const predict=async(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions'])=>{
    const key=JSON.stringify([ids,edits]);if(cache.has(key))return cache.get(key)!;
    const base=await engine.runBaseline(ids);let changed:string|undefined;
    try{if(edits.length)changed=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;const values=(await engine.inspectActivation(changed??base.runId,{stage:'logits'},{position:ids.length-1})).values;cache.set(key,values);return values;}
    finally{if(changed)engine.releaseRun(changed);engine.releaseRun(base.runId);}
  };
  try{
    expect(reference.modelRevision).toBe(loaded.manifest.modelRevision);expect(reference.cases).toHaveLength(24);
    for(const example of reference.cases){
      const workspace=parentProbabilityWorkspace(example.referenceIndex,example.targetTokenId),snapshot=captureParentStories(workspace,workspace.sharedForks!.at(-1)!);
      const trace=traces.cases.find(row=>row.referenceIndex===example.referenceIndex&&row.childBranch===example.branch&&row.targetTokenId===example.targetTokenId)!;
      const positions=parentPositionChanges(measured(trace,snapshot),example.targetTokenId),ranked=rankParentPositions(positions,example.order as typeof orders[number]);
      expect(ranked.map(p=>p.offset)).toEqual(example.ranked.map(p=>p.offset));ranked.forEach((p,i)=>expect([p.a.difference,p.b.difference]).toEqual([example.ranked[i].a,example.ranked[i].b]));
      const before=structuredClone(snapshot),draft=parentPositionBranchDraft(snapshot,example.branch as 'original'|'edited',ranked[0]);
      expect(draft.sources).toEqual(example.plans.map(p=>p.source));expect(draft.parents).toEqual(snapshot.children.map(n=>n.id));expect(draft.origin).toEqual({kind:'parent-probability',groupId:snapshot.group.id,branch:example.branch,index:example.offset});
      for(const source of draft.sources){const actual=chooseToken(await predict(source.prefix,selectionEdits(source)),source.settings,drawAt(source.settings.seed,source.step-1));expect(actual.tokenId).toBe(source.choice.tokenId);expect(actual.probability).toBeCloseTo(source.choice.probability,5);source.choice=actual;}
      const pair=draft.edits.map(edit=>({...edit,factor:example.strengths==='amplify'?2:edit.factor}));
      const plans=draft.sources.map(source=>({source,edit:pair[0],additionalEdits:[pair[1]],duration:example.duration,maxTokens:3})) as ConstructorParameters<typeof SharedStoryFork>[0],session=new SharedStoryFork(plans,loaded.manifest.config);
      while(!session.finished)await session.step(predict);
      session.frames.forEach((frame,side)=>{for(const branch of ['original','edited'] as const){expect(frame[branch]!.tokenIds).toEqual(example.frames[side][branch].tokenIds);frame[branch]!.choices.forEach((choice,i)=>expect(choice.probability).toBeCloseTo(example.frames[side][branch].choices[i].probability,5));}});
      expect(snapshot).toEqual(before);
    }
  }finally{engine.dispose();}expect(tf.memory().numTensors).toBe(memory);
},120000);
it('rejects stale prefixes, choices, target tokens, edit expiry, sampling counters and saved pair identities before another branch',()=>{
  const workspace=parentProbabilityWorkspace(),snapshot=captureParentStories(workspace,workspace.sharedForks!.at(-1)!);
  const example=traces.cases.find(row=>row.referenceIndex===6&&row.childBranch==='edited'&&row.targetTokenId===471)!;
  const point=rankParentPositions(parentPositionChanges(measured(example,snapshot),471),'absolute')[0];
  for(const mutate of [
    (p:typeof point)=>{p.a.child.source.prefix[0]=33;},(p:typeof point)=>{p.b.child.source.choice.probability*=.5;},
    (p:typeof point)=>{p.b.child.targetTokenId=1;},(p:typeof point)=>{p.a.parent.source.prefix[0]=33;},
    (p:typeof point)=>{p.b.child.source.step++;},(p:typeof point)=>{p.a.child.source.settings.seed++;},
    (p:typeof point)=>{p.a.child.source.continuationEdits=[];},(p:typeof point)=>{p.a.difference=0;},
  ]){const changed=structuredClone(point);mutate(changed);expect(()=>parentPositionBranchDraft(snapshot,'edited',changed)).toThrow('no longer matches');}
  const bad=structuredClone(snapshot);bad.group.nodeIds[0]='another-comparison';expect(()=>parentPositionBranchDraft(bad,'edited',point)).toThrow('another comparison');
  const pair=structuredClone(snapshot);pair.children[1].plan.edit.factor=2;expect(()=>parentPositionBranchDraft(pair,'edited',point)).toThrow('same selected pair');
  const original=structuredClone(snapshot),draft=parentPositionBranchDraft(snapshot,'edited',point);draft.sources[0].prefix[0]=33;draft.edits[0].factor=2;expect(snapshot).toEqual(original);
});
