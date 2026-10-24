import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { captureParentStories, parentCounts, parentDifferences, parentSelection, traceParentProbabilities, type ParentProbabilityRows } from '../../src/story-parent-probabilities';
import { parseWorkspace } from '../../src/story-workspace';
import { selectForkToken } from '../../src/story-fork';
import { chooseToken, drawAt, selectionEdits } from '../../src/continuation';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import { parentProbabilityWorkspace } from '../support/parent-probability-workspace';
import reference from '../fixtures/parent-probabilities-reference.json';
beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
const empty=():ParentProbabilityRows=>[{parent:[],child:[]},{parent:[],child:[]}];
it('measures actual parent suffixes and children against 68 independent PyTorch traces, including root parents and both stopping lengths',async()=>{
  const memory=tf.memory().numTensors,loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),cache=new Map<string,ArrayLike<number>>();
  const predict=async(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions'])=>{
    const key=JSON.stringify([ids,edits]);if(cache.has(key))return cache.get(key)!;
    const baseline=await engine.runBaseline(ids);let modified:string|undefined;
    try{if(edits.length)modified=(await engine.compare({baselineRunId:baseline.runId,interventions:edits})).modifiedRunId;const values=(await engine.inspectActivation(modified??baseline.runId,{stage:'logits'},{position:ids.length-1})).values;cache.set(key,values);return values;}
    finally{if(modified)engine.releaseRun(modified);engine.releaseRun(baseline.runId);}
  };
  try{
    expect(reference.modelRevision).toBe(loaded.manifest.modelRevision);
    for(const example of reference.cases){
      const workspace=parentProbabilityWorkspace(example.referenceIndex,example.targetTokenId);expect(parseWorkspace(workspace,loaded.manifest)).toEqual(workspace);
      const snapshot=captureParentStories(workspace,workspace.sharedForks!.at(-1)!),rows=empty();
      // Saved CPU choices must replay at the existing strict tolerance; PyTorch remains the independent probability oracle.
      for(const side of [0,1])for(const role of ['parent','child'] as const){
        const count=parentCounts(snapshot,example.childBranch as 'original'|'edited')[side][role];
        for(let offset=0;offset<count;offset++){
          const source=role==='parent'?parentSelection(snapshot,side,offset):selectForkToken(snapshot.children[side].plan,snapshot.children[side].frame,example.childBranch as 'original'|'edited',offset);
          const actual=chooseToken(await predict(source.prefix,selectionEdits(source)),source.settings,drawAt(source.settings.seed,source.step-1));expect(actual.tokenId).toBe(source.choice.tokenId);expect(actual.probability).toBeCloseTo(source.choice.probability,5);
          if(role==='child')snapshot.children[side].frame[example.childBranch as 'original'|'edited']!.choices[offset]=actual;
          else{const parent=snapshot.parents[side];if(parent.kind==='node')parent.node.frame[source.branch]!.choices[source.step-parent.node.plan.source.step]=actual;else parent.root.frame[source.branch]!.choices[source.step-1]=actual;}
        }
        snapshot.children[side].plan.source.choice=parentSelection(snapshot,side,0).choice;
      }
      const before=structuredClone(snapshot);
      await traceParentProbabilities(snapshot,example.childBranch as 'original'|'edited',example.targetTokenId,loaded.manifest.config,predict,(side,role,point)=>rows[side][role].push(point),()=>false);
      for(const side of [0,1])for(const role of ['parent','child'] as const){
        const expected=example.rows[side][role];expect(rows[side][role]).toHaveLength(expected.length);
        rows[side][role].forEach((point,i)=>{const r=expected[i];expect([point.offset,point.source.step,point.source.choice.tokenId,point.targetTokenId]).toEqual([r.offset,r.step,r.tokenId,r.targetTokenId]);expect(point.source.prefix).toEqual(r.prefix);expect(selectionEdits(point.source)).toEqual(r.edits);expect(point.source.choice.probability).toBeCloseTo(r.recordedProbability,5);expect(point.probability).toBeCloseTo(r.probability,5);});
      }
      const differences=parentDifferences(rows,example.targetTokenId);
      differences.forEach((track,side)=>{expect(track).toHaveLength(Math.min(example.rows[side].parent.length,example.rows[side].child.length));track.forEach(point=>expect(point.difference).toBe(point.child.probability-point.parent.probability));});
      if(example.childBranch==='original')expect(differences.flat().every(row=>row.samePrefix&&row.difference===0)).toBe(true);
      expect(snapshot).toEqual(before);
    }
  }finally{engine.dispose();}expect(tf.memory().numTensors).toBe(memory);
},120000);
it('validates target and ancestry and excludes missing or mismatched comparison points',async()=>{
  const loaded=loadShippedArtifact('tinystories'),workspace=parentProbabilityWorkspace(),snapshot=captureParentStories(workspace,workspace.sharedForks!.at(-1)!);
  const rows=empty();let calls=0;
  // Returning real stored source logits is exercised above; cancellation before any prediction emits nothing.
  await traceParentProbabilities(snapshot,'edited',471,loaded.manifest.config,async()=>{throw new Error('must not infer');},(side,role,point)=>rows[side][role].push(point),()=>true);expect(rows).toEqual(empty());
  for(const target of [0,2,-1,loaded.manifest.config.vocabularySize,1.5,NaN])await expect(traceParentProbabilities(snapshot,'edited',target,loaded.manifest.config,async()=>{calls++;return [];},()=>{},()=>false)).rejects.toThrow('eligible');expect(calls).toBe(0);
  const bad=structuredClone(workspace);bad.nodes.at(-1)!.plan.source.prefix[0]=33;expect(()=>captureParentStories(bad,bad.sharedForks!.at(-1)!)).toThrow('actual parent');
  const first=parentSelection(snapshot,0,0),child=selectForkToken(snapshot.children[0].plan,snapshot.children[0].frame,'edited',0);
  expect(first.prefix).toEqual(child.prefix);expect(parentCounts(snapshot,'edited')[0].parent).toBe(4);
  // A mismatched target or global position cannot become a difference, and a measured zero remains present.
  const p={offset:0,targetTokenId:471,probability:.2,source:first};rows[0].parent=[p];rows[0].child=[{...p,source:child}];expect(parentDifferences(rows,471)[0][0].difference).toBe(0);
  rows[0].child[0].targetTokenId=1;expect(parentDifferences(rows,471)[0]).toEqual([]);
  rows[0].child=[{...p,source:{...child,step:child.step+1}}];expect(parentDifferences(rows,471)[0]).toEqual([]);
});

it('snapshots inputs, publishes only complete predictions on cancellation, and replay-checks actual choices for another target',async()=>{
  const loaded=loadShippedArtifact('tinystories'),workspace=parentProbabilityWorkspace(),snapshot=captureParentStories(workspace,workspace.sharedForks!.at(-1)!),engine=new RunEngine(loaded.artifact);
  const predict=async(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions'])=>{const base=await engine.runBaseline(ids);let modified:string|undefined;try{if(edits.length)modified=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;return (await engine.inspectActivation(modified??base.runId,{stage:'logits'},{position:ids.length-1})).values;}finally{if(modified)engine.releaseRun(modified);engine.releaseRun(base.runId);}};
  try{
    const rows=empty();let calls=0;
    await traceParentProbabilities(snapshot,'edited',471,loaded.manifest.config,async(ids,edits)=>{snapshot.children[1].frame.edited!.tokenIds[0]=33;calls++;expect(ids[0]).toBe(0);return predict(ids,edits);},(side,role,point)=>rows[side][role].push(point),()=>calls>=4);
    expect(rows[0].parent).toHaveLength(1);expect(rows[0].child).toHaveLength(1);expect(rows[1].parent).toHaveLength(1);expect(rows[1].child).toHaveLength(0);
    const fresh=captureParentStories(workspace,workspace.sharedForks!.at(-1)!),parent=fresh.parents[0];if(parent.kind!=='node')throw new Error('Expected a parent node.');
    const index=fresh.children[0].plan.source.step-parent.node.plan.source.step;parent.node.frame[fresh.children[0].plan.source.branch]!.choices[index].probability*=.5;
    await expect(traceParentProbabilities(fresh,'edited',1,loaded.manifest.config,predict,()=>{},()=>false)).rejects.toThrow('reproduced');
  }finally{engine.dispose();}
});
