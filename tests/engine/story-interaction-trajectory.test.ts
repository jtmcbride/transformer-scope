import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, type StoryFrame } from '../../src/continuation';
import { forkEdits, type ForkPlan } from '../../src/story-fork';
import { interactionTrend, traceStoryInteractions, type InteractionPoint, type InteractionTracks } from '../../src/story-interaction-trajectory';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import shared from '../fixtures/story-shared-forks-reference.json';
import reference from '../fixtures/story-interaction-trajectories-reference.json';
beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
const config=loadShippedArtifact('tinystories').manifest.config;
it('measures both targets on their own exact prefixes against independent PyTorch along all four shared scenarios',async()=>{
  const loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),memory=tf.memory().numTensors;
  expect(reference.modelRevision).toBe(loaded.manifest.modelRevision);
  async function predict(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions']){
    const base=await engine.runBaseline(ids);let modified:string|undefined;
    try{if(edits.length)modified=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;return (await engine.inspectActivation(modified??base.runId,{stage:'logits'},{position:ids.length-1})).values;}
    finally{if(modified)engine.releaseRun(modified);engine.releaseRun(base.runId);}
  }
  try{
    for(const [caseIndex,example] of shared.cases.entries()){
      const plans=structuredClone(example.plans) as [ForkPlan,ForkPlan],frames=structuredClone(example.frames) as [StoryFrame,StoryFrame];
      for(const side of [0,1])for(const branch of ['original','edited'] as const)for(const [i,expected] of frames[side][branch]!.choices.entries()){
        const ids=frames[side][branch]!.tokenIds.slice(0,plans[side].source.prefix.length+i),logits=await predict(ids,forkEdits(plans[side],branch==='edited',i));
        const choice=chooseToken(logits,plans[side].source.settings,drawAt(plans[side].source.settings.seed,plans[side].source.step-1+i));
        expect(choice.tokenId).toBe(expected.tokenId);expect(choice.probability).toBeCloseTo(expected.probability,5);frames[side][branch]!.choices[i]=choice;
      }
      const snapshot=structuredClone({plans,frames});
      for(const branch of ['original','edited'] as const){
        const rows:InteractionTracks=[[],[]],order:number[]=[];
        await traceStoryInteractions(plans,frames,branch,config,async(source,use)=>{await use(async(ids,edits)=>{expect(ids).toEqual(source.prefix);return predict(ids,edits);});},(side,row)=>{rows[side].push(row);order.push(row.index);},()=>false);
        expect(order).toEqual([...order].sort((a,b)=>a-b));
        for(const side of [0,1]){
          const expected=reference.cases[caseIndex][branch][side];expect(rows[side]).toHaveLength(expected.length);
          rows[side].forEach((row,i)=>{const r=expected[i];expect([row.index,row.step,row.tokenId,row.active]).toEqual([r.index,r.step,r.tokenId,r.active]);for(const field of ['before','after'] as const)expect(row[field]).toBeCloseTo(r[field],5);row.individualProbabilities.forEach((p,j)=>expect(p).toBeCloseTo(r.individualProbabilities[j],5));expect(Math.abs(row.interaction-r.interaction)).toBeLessThanOrEqual(4*5e-6);if(!row.active)expect(row.interaction).toBe(0);});
        }
        expect(tf.memory().numTensors).toBe(memory);
      }
      expect({plans,frames}).toEqual(snapshot);
    }
  }finally{engine.dispose();}
},120000);
function synthetic(){
  const plans=structuredClone(shared.cases[0].plans) as [ForkPlan,ForkPlan],logits=[-99,-99,-99,0,.1,.2];
  const frames=plans.map(plan=>{plan.source.settings={temperature:0,topP:.9,seed:42,maxTokens:6};const choice=chooseToken(logits,plan.source.settings,0);return {step:2,original:{tokenIds:[...plan.source.prefix,choice.tokenId,choice.tokenId],choices:[choice,choice]},edited:{tokenIds:[...plan.source.prefix,choice.tokenId,choice.tokenId],choices:[choice,choice]}};}) as [StoryFrame,StoryFrame];
  return {plans,frames,logits};
}
it('publishes only complete replayed points, snapshots caller data and never fills missing measurements with zero',async()=>{
  const {plans,frames,logits}=synthetic(),rows:InteractionTracks=[[],[]],original=structuredClone({plans,frames});let scopes=0;
  await traceStoryInteractions(plans,frames,'edited',config,async(source,use)=>{expect(source.prefix).toEqual(original.frames[scopes%2].edited!.tokenIds.slice(0,original.plans[scopes%2].source.prefix.length+Math.floor(scopes/2)));scopes++;frames[1].edited!.tokenIds[0]=33;plans[1].source.step=25;await use(async()=>logits);},(side,row)=>rows[side].push(row),()=>false);
  expect(rows.map(track=>track.length)).toEqual([2,2]);expect(rows.flat().every(row=>row.interaction===0)).toBe(true);expect(scopes).toBe(4);
  const fresh=synthetic();let calls=0;const partial:InteractionTracks=[[],[]];
  await traceStoryInteractions(fresh.plans,fresh.frames,'edited',config,async(_,use)=>{await use(async()=>{calls++;return logits;});},(side,row)=>partial[side].push(row),()=>calls>=6);
  expect(partial[0]).toHaveLength(1);expect(partial[1]).toHaveLength(0);
});
it('checks the actual edited-path replay, rejects changed choices and retains earlier points on a later source failure',async()=>{
  const {plans,frames,logits}=synthetic();frames[1].edited!.choices[0].probability=.1;const rows:InteractionTracks=[[],[]];
  await expect(traceStoryInteractions(plans,frames,'edited',config,async(_,use)=>{await use(async()=>logits);},(side,row)=>rows[side].push(row),()=>false)).rejects.toThrow('reproduced');
  expect(rows[0]).toHaveLength(1);expect(rows[1]).toHaveLength(0);
  const fresh=synthetic();let calls=0;
  await expect(traceStoryInteractions(fresh.plans,fresh.frames,'edited',config,async(_,use)=>{await use(async()=>{if(++calls===6)throw new Error('single failed');return logits;});},()=>{},()=>false)).rejects.toThrow('single failed');
});
it('labels growing, fading and reversing magnitude at displayed precision without bridging unmeasured offsets',()=>{
  const point=(index:number,interaction:number)=>({index,interaction}) as InteractionPoint;
  expect(interactionTrend(point(0,.1))).toBe('Starting point');expect(interactionTrend(point(1,.2),point(0,.1))).toBe('Growing');expect(interactionTrend(point(1,.05),point(0,.1))).toBe('Fading');
  expect(interactionTrend(point(1,-.02),point(0,.1))).toBe('Reversed');expect(interactionTrend(point(1,0),point(0,.1))).toBe('Fading to zero');expect(interactionTrend(point(1,1e-10),point(0,-1e-10))).toBe('Near zero');
  expect(interactionTrend(point(1,.1),point(0,.1))).toBe('Steady');expect(interactionTrend(point(3,-.1),point(0,.1))).toBe('Starting point');
});
