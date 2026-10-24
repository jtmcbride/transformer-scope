import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { trajectoryDifferences, rankTrajectoryDifferences, trajectoryBranchDraft } from '../../src/story-trajectory-differences';
import type { InteractionPoint, InteractionTracks } from '../../src/story-interaction-trajectory';
import type { StoryNode, SharedStoryGroup } from '../../src/story-workspace';
import { RunEngine } from '../../src/engine/runs';
import { SharedStoryFork } from '../../src/story-shared-fork';
import { chooseToken, drawAt } from '../../src/continuation';
import { loadShippedArtifact } from './retrieval-artifact';
import input from '../fixtures/story-trajectory-workspace.json';
import fixed from '../fixtures/story-fixed-token-trajectories-reference.json';
import reference from '../fixtures/trajectory-rebranches-reference.json';

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
const point=(index:number,delta:number,targetTokenId=471)=>({index,delta,targetTokenId}) as InteractionPoint;
it('matches offsets and fixed targets, keeps measured zeroes, and ranks raw signed gaps with deterministic ties',()=>{
  const tracks:InteractionTracks=[[point(4,.1),point(0,.2),point(1,0),point(2,-.2),point(3,.1)], [point(3,.4,1),point(1,0),point(2,-.1),point(0,.1)]];
  const snapshot=structuredClone(tracks),rows=trajectoryDifferences(tracks,471);
  expect(rows.map(row=>[row.index,row.difference])).toEqual([[0,-.1],[1,0],[2,.1]]);
  expect(rankTrajectoryDifferences(rows,'absolute').map(row=>row.index)).toEqual([0,2,1]);
  expect(rankTrajectoryDifferences(rows,'increase').map(row=>row.index)).toEqual([2,1,0]);
  expect(rankTrajectoryDifferences(rows,'decrease').map(row=>row.index)).toEqual([0,1,2]);
  expect(trajectoryDifferences([tracks[0],[]],471)).toEqual([]);expect(tracks).toEqual(snapshot);
  const tiny=trajectoryDifferences([[point(0,0),point(1,0)],[point(0,1e-10),point(1,2e-10)]],471);
  expect(rankTrajectoryDifferences(tiny,'absolute').map(row=>row.index)).toEqual([1,0]);
});
it('ranks every independent fixed-target reference on both paths without conflating probability and interaction',()=>{
  for(const target of fixed.targets)for(const example of target.cases)for(const branch of ['original','edited'] as const){
    const tracks=example[branch].map(track=>track.map(row=>({...row,targetTokenId:target.tokenId,alternatives:[]}))) as unknown as InteractionTracks;
    const rows=trajectoryDifferences(tracks,target.tokenId);expect(rows).toHaveLength(Math.min(...tracks.map(track=>track.length)));
    rows.forEach(row=>expect(row.difference).toBe(tracks[1][row.index].delta-tracks[0][row.index].delta));
    expect(rankTrajectoryDifferences(rows,'absolute')[0]?.index).toBe([...rows].sort((a,b)=>Math.abs(b.difference)-Math.abs(a.difference)||a.index-b.index)[0]?.index);
  }
});
it('rebranches strongest offsets against 16 independent four-path PyTorch continuations, retaining actual inherited policies',async()=>{
  const memory=tf.memory().numTensors,loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),cache=new Map<string,ArrayLike<number>>();
  const predict=async(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions'])=>{
    const key=JSON.stringify([ids,edits]);if(cache.has(key))return cache.get(key)!;
    const base=await engine.runBaseline(ids);let changed:string|undefined;
    try{if(edits.length)changed=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;const logits=(await engine.inspectActivation(changed??base.runId,{stage:'logits'},{position:ids.length-1})).values;cache.set(key,logits);return logits;}
    finally{if(changed)engine.releaseRun(changed);engine.releaseRun(base.runId);}
  };
  try{
    expect(reference.modelRevision).toBe(loaded.manifest.modelRevision);
    for(const example of reference.cases){
      const group=input.workspace.sharedForks.find(group=>group.id===example.groupId)! as SharedStoryGroup;
      const nodes=structuredClone(group.nodeIds.map(id=>input.workspace.nodes.find(node=>node.id===id)!)) as [StoryNode,StoryNode];
      const tracks=fixed.targets[0].cases[example.caseIndex][example.branch as 'original'|'edited'].map((track,side)=>track.map(row=>({...row,recordedProbability:nodes[side].frame[example.branch as 'original'|'edited']!.choices[row.index].probability,targetTokenId:example.targetTokenId,alternatives:[]}))) as unknown as InteractionTracks;
      const ranked=rankTrajectoryDifferences(trajectoryDifferences(tracks,example.targetTokenId),'absolute');expect(ranked.map(row=>row.index)).toEqual(example.ranked.map(row=>row.index));
      const snapshot=structuredClone({group,nodes,ranked}),draft=trajectoryBranchDraft(group,nodes,example.branch as 'original'|'edited',ranked[0]);
      expect(draft.sources).toEqual(example.plans.map(plan=>plan.source));expect(draft.parents).toEqual(example.nodeIds);expect(draft.origin.index).toBe(example.index);
      for(const source of draft.sources)source.choice=chooseToken(await predict(source.prefix,source.edits!),source.settings,drawAt(source.settings.seed,source.step-1));
      const plans=example.plans.map((plan,i)=>({...plan,source:draft.sources[i]})) as ConstructorParameters<typeof SharedStoryFork>[0],session=new SharedStoryFork(plans,loaded.manifest.config);
      while(!session.finished)await session.step(predict);
      session.frames.forEach((frame,i)=>{for(const branch of ['original','edited'] as const){expect(frame[branch]!.tokenIds).toEqual(example.frames[i][branch].tokenIds);frame[branch]!.choices.forEach((choice,j)=>expect(choice.probability).toBeCloseTo(example.frames[i][branch].choices[j].probability,5));}});
      expect({group,nodes,ranked}).toEqual(snapshot);
      const bad=structuredClone(ranked[0]);bad.a.tokenId=-1;expect(()=>trajectoryBranchDraft(group,nodes,example.branch as 'original'|'edited',bad)).toThrow('no longer matches');
    }
  }finally{engine.dispose();}expect(tf.memory().numTensors).toBe(memory);
},120000);
