import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { scanStoryHeatmaps, compareInteractions, rankInteractionChanges } from '../../src/story-heatmap-comparison';
import type { HeadPairScanRow } from '../../src/story-head-pair-scan';
import { traceStoryInteractions, type InteractionTracks } from '../../src/story-interaction-trajectory';
import { SharedStoryFork } from '../../src/story-shared-fork';
import type { ForkPlan } from '../../src/story-fork';
import { RunEngine } from '../../src/engine/runs';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/chosen-token-scan-comparison-reference.json';
beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
it('compares 480 same-target pair effects, four-path branches and 192 trajectory points against independent PyTorch',async()=>{
  const memory=tf.memory().numTensors,loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact);
  const sources=structuredClone(reference.sources) as [StorySelection,StorySelection], cache=new Map<string,ArrayLike<number>>();
  const predict=async(ids:number[],edits:ReturnType<typeof selectionEdits>)=>{
    const key=JSON.stringify([ids,edits]),saved=cache.get(key);if(saved)return saved;
    const base=await engine.runBaseline(ids);let changed:string|undefined;
    try{if(edits.length)changed=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;
      const values=(await engine.inspectActivation(changed??base.runId,{stage:'logits'},{position:ids.length-1})).values;
      cache.set(key,values);return values;
    }finally{if(changed)engine.releaseRun(changed);engine.releaseRun(base.runId);}
  };
  try{
    expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
    for(const source of sources)source.choice=chooseToken(await predict(source.prefix,selectionEdits(source)),source.settings,drawAt(source.settings.seed,source.step-1));
    const snapshot=structuredClone(sources);
    for(const example of reference.cases){
      const rows:[HeadPairScanRow[],HeadPairScanRow[]]=[[],[]];
      await scanStoryHeatmaps(sources,loaded.manifest.config,'mute',async(source,use)=>{await use((ids,edits)=>{expect(ids).toEqual(source.prefix);return predict(ids,edits);});},(side,row)=>rows[side].push(row),()=>false,example.targetTokenId);
      expect(rows.map(side=>side.length)).toEqual([120,120]);
      rows.forEach((side,i)=>side.forEach((row,j)=>{const expected=example.rows[i][j];expect(row.edits).toEqual(expected.edits);expect(row.rank).toBe(expected.rank);expect(row.choice).toBe(expected.choice);
        for(const field of ['before','after'] as const)expect(Math.abs(row[field]-expected[field])).toBeLessThanOrEqual(Math.min(5e-6,1e-10+Math.abs(expected[field])*1e-4));
        row.individualProbabilities.forEach((p,k)=>expect(p).toBeCloseTo(expected.individualProbabilities[k],5));
        const probabilityError=Math.abs(row.before-expected.before)+Math.abs(row.after-expected.after);
        expect(row.delta).toBe(row.after-row.before);expect(Math.abs(row.delta-expected.delta)).toBeLessThanOrEqual(probabilityError+1e-12);
        const interactionError=probabilityError+row.individualProbabilities.reduce((sum,p,k)=>sum+Math.abs(p-expected.individualProbabilities[k]),0);
        expect(Math.abs(row.interaction-expected.interaction)).toBeLessThanOrEqual(interactionError+1e-12);
      }));
      const changes=compareInteractions(rows[0],rows[1],'probability');
      changes.forEach((row,i)=>{
        const budget=[0,1].reduce((sum,side)=>sum+Math.abs(rows[side][i].before-example.rows[side][i].before)+Math.abs(rows[side][i].after-example.rows[side][i].after),0);
        expect(row.difference).toBe(row.b.delta-row.a.delta);expect(Math.abs(row.difference-example.changes[i].difference)).toBeLessThanOrEqual(budget+1e-12);
      });
      expect(compareInteractions(rows[0],rows[0],'probability').every(row=>row.difference===0)).toBe(true);
      compareInteractions(rows[1],rows[0],'probability').forEach((row,i)=>expect(row.difference).toBeCloseTo(-changes[i].difference,14));
      expect(compareInteractions(rows[0],rows[1].slice(0,3),'probability')).toHaveLength(3);
      for(const order of ['absolute','increase','decrease'] as const){
        const best=rankInteractionChanges(changes,order)[0];
        for(const duration of ['continuation','prediction'] as const){
          const expected=example.branches[order][duration];expect(best.a.edits).toEqual([expected.plans[0].edit,...expected.plans[0].additionalEdits]);
          const plans=expected.plans.map((plan,i)=>({...plan,source:sources[i]})) as [ForkPlan,ForkPlan],session=new SharedStoryFork(plans,loaded.manifest.config);
          while(!session.finished)await session.step(predict);
          session.frames.forEach((frame,i)=>{for(const branch of ['original','edited'] as const){expect(frame[branch]!.tokenIds).toEqual(expected.frames[i][branch].tokenIds);frame[branch]!.choices.forEach((choice,j)=>expect(choice.probability).toBeCloseTo(expected.frames[i][branch].choices[j].probability,5));}});
          for(const branch of ['original','edited'] as const){
            const tracks:InteractionTracks=[[],[]];
            await traceStoryInteractions(session.plans,session.frames,branch,loaded.manifest.config,async(_,use)=>{await use(predict);},(side,row)=>tracks[side].push(row),()=>false,{kind:'fixed',tokenId:example.targetTokenId});
            for(const side of [0,1])tracks[side].forEach((row,i)=>{const r=expected.tracks[branch][side][i];
              expect([row.index,row.step,row.tokenId,row.targetTokenId,row.active]).toEqual([r.index,r.step,r.tokenId,r.targetTokenId,r.active]);
              for(const field of ['before','after','delta'] as const)expect(row[field]).toBeCloseTo(r[field],5);
              expect(row.delta).toBe(row.after-row.before);if(!row.active)expect(row.delta).toBe(0);
            });
          }

        }
      }
      expect(sources).toEqual(snapshot);
    }
  }finally{engine.dispose();}
  expect(tf.memory().numTensors).toBe(memory);
},120000);
it('retains actual-choice replay and suppresses incomplete second-story results for fixed targets',async()=>{
  const config=loadShippedArtifact('tinystories').manifest.config, sources=structuredClone(reference.sources) as [StorySelection,StorySelection];
  const logits=Array.from({length:config.vocabularySize},(_,id)=>id===3?1:0);
  for(const source of sources)source.choice=chooseToken(logits,source.settings,drawAt(source.settings.seed,source.step-1));
  const rows:[HeadPairScanRow[],HeadPairScanRow[]]=[[],[]];let stop=false,side=0,calls=0;
  await scanStoryHeatmaps(sources,config,'mute',async(_,use)=>{const current=side++;await use(async()=>{if(current===1&&++calls===4)stop=true;return logits;});},(i,row)=>rows[i].push(row),()=>stop,471);
  expect(rows.map(side=>side.length)).toEqual([120,0]);expect(compareInteractions(...rows,'probability')).toEqual([]);
  sources[1].choice.probability+=.01;
  await expect(scanStoryHeatmaps(sources,config,'mute',async(_,use)=>use(async()=>logits),()=>{},()=>false,471)).rejects.toThrow('reproduce');
});
