import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, type StoryFrame } from '../../src/continuation';
import { proposedEdits, selectForkToken, type ForkPlan } from '../../src/story-fork';
import { SharedStoryFork } from '../../src/story-shared-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-shared-forks-reference.json';
beforeAll(async () => {await tf.setBackend('cpu');await tf.ready();});
const config = loadShippedArtifact('tinystories').manifest.config;
function synthetic() {
  const plans=structuredClone(reference.cases[0].plans) as [ForkPlan,ForkPlan],logits=[-99,-99,-99,0,.1,.2];
  plans.forEach((plan,i)=>{plan.source.step+=i*2;plan.source.settings={temperature:.8,topP:.9,seed:i?97:23,maxTokens:6};plan.source.choice=chooseToken(logits,plan.source.settings,drawAt(plan.source.settings.seed,plan.source.step-1));});
  return {plans,logits};
}
it('reproduces four independent PyTorch continuations for sustained, temporary, inherited and identity pairs without retained tensors',async()=>{
  const loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),memory=tf.memory().numTensors;
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  async function predict(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions']) {
    const base=await engine.runBaseline(ids);let modified:string|undefined;
    try {if(edits.length)modified=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;
      return (await engine.inspectActivation(modified??base.runId,{stage:'logits'},{position:ids.length-1})).values;
    } finally {if(modified)engine.releaseRun(modified);engine.releaseRun(base.runId);}
  }
  try {
    for(const example of reference.cases) {
      const plans=structuredClone(example.plans) as [ForkPlan,ForkPlan];
      // Replay uses CPU-recorded probabilities; cross-framework comparisons keep their existing precision.
      for(const plan of plans){const logits=await predict(plan.source.prefix,plan.source.edits??[]);const choice=chooseToken(logits,plan.source.settings,drawAt(plan.source.settings.seed,plan.source.step-1));expect(choice.tokenId).toBe(plan.source.choice.tokenId);expect(choice.probability).toBeCloseTo(plan.source.choice.probability,5);plan.source.choice=choice;}
      const before=structuredClone(plans),shared=new SharedStoryFork(plans,config);
      while(!shared.finished)await shared.step(predict);
      shared.frames.forEach((frame,side)=>{expect(frame.step).toBe(example.frames[side].step);for(const branch of ['original','edited'] as const){expect(frame[branch]!.tokenIds).toEqual(example.frames[side][branch].tokenIds);expect(frame[branch]!.finish).toBe(example.frames[side][branch].finish);frame[branch]!.choices.forEach((choice,i)=>expect(choice.probability).toBeCloseTo(example.frames[side][branch].choices[i].probability,5));}
        expect(selectForkToken(shared.plans[side],frame,'edited',0).edits).toContainEqual(shared.plans[side].edit);
      });
      expect(plans).toEqual(before);expect(tf.memory().numTensors).toBe(memory);
      const resumed=new SharedStoryFork(plans,config);resumed.restore(shared.frames);expect(resumed.frames).toEqual(shared.frames);expect(resumed.finished).toBe(true);
    }
  }finally{engine.dispose();}
},60000);
it('snapshots both sources and requires the same exact two-head proposal, duration and budget',()=>{
  const {plans}=synthetic(),saved=structuredClone(plans),shared=new SharedStoryFork(plans,config);
  plans[1].source.prefix[0]=33;plans[0].edit.factor=2;expect(shared.plans).toEqual(saved);
  for(const corrupt of [(p:ForkPlan)=>p.edit.factor=2,(p:ForkPlan)=>p.duration='prediction',(p:ForkPlan)=>p.maxTokens=5,(p:ForkPlan)=>delete p.additionalEdits]){const pair=structuredClone(saved);corrupt(pair[1]);expect(()=>new SharedStoryFork(pair,config)).toThrow();}
  const frame=shared.frames;frame[0].original.tokenIds[0]=33;expect(shared.frames[0].original.tokenIds[0]).toBe(0);
});
it('publishes neither source on second-source failure, cancellation, or replay mismatch and resumes exact seeded draws',async()=>{
  const {plans,logits}=synthetic(),shared=new SharedStoryFork(plans,config),before=shared.frames;let calls=0;
  await expect(shared.step(async()=>{if(++calls===4)throw new Error('B failed');return logits;})).rejects.toThrow('B failed');expect(shared.frames).toEqual(before);
  calls=0;expect(await shared.step(async()=>{calls++;return logits;},()=>calls>=3)).toBeNull();expect(shared.frames).toEqual(before);
  const wrong=structuredClone(plans);wrong[1].source.choice.probability=0;
  const failed=new SharedStoryFork(wrong,config);await expect(failed.step(async()=>logits)).rejects.toThrow('reproduce');expect(failed.frames).toEqual(before);
  await shared.step(async()=>logits);const resumed=new SharedStoryFork(plans,config);resumed.restore(shared.frames);
  while(!shared.finished){await shared.step(async()=>logits);await resumed.step(async()=>logits);}
  expect(resumed.frames).toEqual(shared.frames);
  shared.frames.forEach((frame,side)=>{expect(frame.edited).toEqual(frame.original);frame.original.choices.forEach((choice,i)=>expect(choice).toEqual(chooseToken(logits,plans[side].source.settings,drawAt(plans[side].source.settings.seed,plans[side].source.step-1+i))));expect(frame.original.choices[0]).toEqual(plans[side].source.choice);expect(proposedEdits(shared.plans[side])).toEqual(proposedEdits(plans[0]));});
});
it('retains an ended source while the other continues, and rejects mismatched paused shared frames',async()=>{
  const {plans,logits}=synthetic();plans[0].source.choice=chooseToken([-99,4,-99,0,.1,.2],plans[0].source.settings,drawAt(23,plans[0].source.step-1));
  const shared=new SharedStoryFork(plans,config);
  await shared.step(async ids=>ids.length===plans[0].source.prefix.length&&ids.every((id,i)=>id===plans[0].source.prefix[i])?[-99,4,-99,0,.1,.2]:logits);
  // Both sources in this fixture have different exact prefixes.
  expect(shared.frames[0].original.finish).toBe('end of story');
  const ended=shared.frames[0];await shared.step(async()=>logits);expect(shared.frames[0]).toEqual(ended);expect(shared.frames[1].step).toBe(2);
  const bad=shared.frames;bad[1]={...beforeFrame(plans[1])};expect(()=>new SharedStoryFork(plans,config).restore(bad)).toThrow('same generation step');
  function beforeFrame(plan:ForkPlan):StoryFrame{return {step:0,original:{tokenIds:[...plan.source.prefix],choices:[]},edited:{tokenIds:[...plan.source.prefix],choices:[]}};}
});
