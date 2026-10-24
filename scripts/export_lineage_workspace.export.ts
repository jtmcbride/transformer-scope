// Real CPU-recorded lifecycle input. Probability expectations remain in the separate PyTorch exporter.
import { readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../src/engine/runs';
import { loadShippedArtifact } from '../tests/engine/retrieval-artifact';
import { selectStoryToken } from '../src/continuation';
import { StoryFork, selectForkToken, type ForkPlan } from '../src/story-fork';
import { SharedStoryFork } from '../src/story-shared-fork';
import { emptyWorkspace, parseWorkspace } from '../src/story-workspace';
import type { HeadScaleIntervention } from '../src/engine/contracts';
const read=(name:string)=>JSON.parse(readFileSync(new URL(`../tests/fixtures/${name}`,import.meta.url),'utf8'));
it('exports validated CPU-recorded original, parent and latest histories with a shorter original tail',async()=>{
  await tf.setBackend('cpu');await tf.ready();const memory=tf.memory().numTensors,loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact);
  try{
    const input=read('story-trajectory-workspace.json'),parentReference=read('parent-probabilities-reference.json').root,latestReference=read('parent-position-rebranches-reference.json').cases[22];
    assert.equal(input.backend,'cpu');assert.equal(input.modelRevision,loaded.manifest.modelRevision);
    const workspace=emptyWorkspace(loaded.manifest);workspace.root=structuredClone(input.workspace.root);const root=workspace.root!;
    async function predict(ids:number[],edits:HeadScaleIntervention[]=[]){const base=await engine.runBaseline(ids);let modified:string|undefined;try{if(edits.length)modified=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;return (await engine.inspectActivation(modified??base.runId,{stage:'logits'},{position:ids.length-1})).values;}finally{if(modified)engine.releaseRun(modified);engine.releaseRun(base.runId);}}
    function check(frame:StoryFork['frame'],expected:any){for(const branch of ['original','edited'] as const){assert.deepEqual(frame[branch]!.tokenIds,expected[branch].tokenIds);frame[branch]!.choices.forEach((choice,i)=>assert.ok(Math.abs(choice.probability-expected[branch].choices[i].probability)<5e-6));}}
    const parents=parentReference.plans.map((plan:ForkPlan,side:number)=>new StoryFork({...plan,source:selectStoryToken(root.frame,root.tokens.length,side?'edited':'original',plan.source.step-1,root.settings,root.edit)},loaded.manifest.config)) as [StoryFork,StoryFork];
    for(const [side,parent] of parents.entries()){while(!parent.finished)await parent.step(predict);check(parent.frame,parentReference.frames[side]);workspace.nodes.push({id:`parent-child-${side}`,parentId:null,plan:parent.plan,frame:parent.frame});}
    workspace.sharedForks!.push({id:'parent-comparison',nodeIds:['parent-child-0','parent-child-1'],labels:['Parent A','Parent B'],targetTokenId:471});
    const plans=latestReference.plans.map((plan:ForkPlan,side:number)=>({...plan,source:selectForkToken(parents[side].plan,parents[side].frame,plan.source.branch,plan.source.step-parents[side].plan.source.step)})) as [ForkPlan,ForkPlan],latest=new SharedStoryFork(plans,loaded.manifest.config);
    while(!latest.finished)await latest.step(predict);
    latest.frames.forEach((frame,side)=>{check(frame,latestReference.frames[side]);workspace.nodes.push({id:`lineage-latest-${side}`,parentId:`parent-child-${side}`,plan:latest.plans[side],frame});});
    workspace.sharedForks!.push({id:'lineage-comparison',nodeIds:['lineage-latest-0','lineage-latest-1'],labels:['Latest A','Latest B'],targetTokenId:471});
    root.frame.step=15;for(const branch of ['original','edited'] as const){const track=root.frame[branch]!;track.choices.splice(15);track.tokenIds.splice(root.tokens.length+15);delete track.finish;}
    const valid=parseWorkspace(workspace,loaded.manifest);writeFileSync(new URL('../tests/fixtures/story-lineage-workspace.json',import.meta.url),JSON.stringify({backend:'cpu',modelRevision:loaded.manifest.modelRevision,rebranchReferenceIndex:22,workspace:valid},null,2)+'\n');
    console.log('Exported a CPU-recorded two-generation workspace with a paused original history and both root paths.');
  }finally{engine.dispose();assert.equal(tf.memory().numTensors,memory);}
},120000);
