// Real CPU-recorded browser inputs; numerical trajectory expectations come from the separate PyTorch exporter.
import { readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { it } from 'vitest';
import { RunEngine } from '../src/engine/runs';
import { loadShippedArtifact } from '../tests/engine/retrieval-artifact';
import * as continuation from '../src/continuation';
import * as forks from '../src/story-fork';
import { SharedStoryFork } from '../src/story-shared-fork';
import * as workspace from '../src/story-workspace';
import type { ForkPlan } from '../src/story-fork';
import type { HeadScaleIntervention } from '../src/engine/contracts';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
const read = (name:string) => JSON.parse(readFileSync(new URL(`../tests/fixtures/${name}`,import.meta.url),'utf8'));
it('exports validated CPU browser inputs',async()=>{
let engine:RunEngine|undefined;
try{
  await tf.setBackend('cpu');await tf.ready();
  const loaded=loadShippedArtifact('tinystories');engine=new RunEngine(loaded.artifact);
  async function predict(ids:number[],edits:HeadScaleIntervention[]=[]){const base=await engine!.runBaseline(ids);let modified;try{if(edits.length)modified=(await engine!.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;return (await engine!.inspectActivation(modified??base.runId,{stage:'logits'},{position:ids.length-1})).values;}finally{if(modified)engine!.releaseRun(modified);engine!.releaseRun(base.runId);}}
  const root=read('continuation-reference.json'),parent=read('story-forks-reference.json').cases[0],shared=read('story-shared-forks-reference.json');
  const session=new continuation.StoryContinuation(root.promptTokenIds,root.settings,64,root.edit);
  while(!session.finished)await session.step((ids,edit)=>predict(ids,edit?[edit]:[]));
  const first=new forks.StoryFork({...parent.plan,source:continuation.selectStoryToken(session.frame,root.promptTokenIds.length,'original',5,root.settings,root.edit)},loaded.manifest.config);
  while(!first.finished)await first.step(predict);
  const data=workspace.emptyWorkspace(loaded.manifest);data.root={prompt:root.prompt,tokens:root.promptTokenIds,settings:root.settings,edit:root.edit,frame:session.frame};data.nodes=[{id:'trajectory-parent',parentId:null,plan:first.plan,frame:first.frame}];
  for(const caseIndex of [0,1]){
    const plans=shared.cases[caseIndex].plans.map((plan:ForkPlan,side:number)=>({...plan,source:forks.selectForkToken(first.plan,first.frame,side?'edited':'original',1)}));
    const pair=new SharedStoryFork(plans as [ForkPlan,ForkPlan],loaded.manifest.config);while(!pair.finished)await pair.step(predict);
    const nodeIds:[string,string]=[`trajectory-${caseIndex}-A`,`trajectory-${caseIndex}-B`];
    pair.frames.forEach((frame,side)=>{for(const branch of ['original','edited'] as const){const expected=shared.cases[caseIndex].frames[side][branch];assert.deepEqual(frame[branch].tokenIds,expected.tokenIds);frame[branch].choices.forEach((choice,i)=>assert.ok(Math.abs(choice.probability-expected.choices[i].probability)<5e-6));}data.nodes.push({id:nodeIds[side],parentId:'trajectory-parent',plan:pair.plans[side],frame});});
    data.sharedForks.push({id:`trajectory-group-${caseIndex}`,nodeIds,labels:['Comparison 1 · Keep source settings','Comparison 1 · With head change']});
  }
  const valid=workspace.parseWorkspace(data,loaded.manifest);
  writeFileSync(new URL('../tests/fixtures/story-trajectory-workspace.json',import.meta.url),JSON.stringify({backend:'cpu',modelRevision:loaded.manifest.modelRevision,workspace:valid},null,2)+'\n');
  console.log('Exported a validated CPU-recorded workspace with sustained and prediction-only shared comparisons.');
}finally{engine?.dispose();}
},120000);
