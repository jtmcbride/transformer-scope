import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { captureLineageStories, lineageCounts, originalDifferences, originalSelection, traceLineageProbabilities, type LineageProbabilityRows, type LineageSnapshot } from '../../src/story-lineage-probabilities';
import { parentDifferences, parentSelection } from '../../src/story-parent-probabilities';
import { selectForkToken } from '../../src/story-fork';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../../src/continuation';
import { RunEngine } from '../../src/engine/runs';
import { parseWorkspace } from '../../src/story-workspace';
import { loadShippedArtifact } from './retrieval-artifact';
import { lineageProbabilityWorkspace } from '../support/lineage-probability-workspace';
import { parentProbabilityWorkspace } from '../support/parent-probability-workspace';
import reference from '../fixtures/lineage-probabilities-reference.json';

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready();});
const empty=():LineageProbabilityRows=>[{original:[],parent:[],child:[]},{original:[],parent:[],child:[]}];
const roles=['parent','child','original'] as const;
function runtime(){
  const loaded=loadShippedArtifact('tinystories'),engine=new RunEngine(loaded.artifact),cache=new Map<string,ArrayLike<number>>();
  const predict=async(ids:number[],edits:Parameters<RunEngine['compare']>[0]['interventions'])=>{const key=JSON.stringify([ids,edits]);if(cache.has(key))return cache.get(key)!;const base=await engine.runBaseline(ids);let changed:string|undefined;
    try{if(edits.length)changed=(await engine.compare({baselineRunId:base.runId,interventions:edits})).modifiedRunId;const values=(await engine.inspectActivation(changed??base.runId,{stage:'logits'},{position:ids.length-1})).values;cache.set(key,values);return values;}
    finally{if(changed)engine.releaseRun(changed);engine.releaseRun(base.runId);}};
  return {loaded,engine,predict};
}
async function cpuChoices(snapshot:LineageSnapshot,branch:StorySelection['branch'],predict:ReturnType<typeof runtime>['predict']){
  const counts=lineageCounts(snapshot,branch);
  for(const side of [0,1])for(const role of roles)for(let offset=0;offset<counts[side][role];offset++){
    const source=role==='original'?originalSelection(snapshot,side,offset):role==='parent'?parentSelection(snapshot,side,offset):selectForkToken(snapshot.children[side].plan,snapshot.children[side].frame,branch,offset);
    const actual=chooseToken(await predict(source.prefix,selectionEdits(source)),source.settings,drawAt(source.settings.seed,source.step-1));expect(actual.tokenId).toBe(source.choice.tokenId);expect(actual.probability).toBeCloseTo(source.choice.probability,5);
    if(role==='original')snapshot.originals[side].root.frame[source.branch]!.choices[source.step-1]=actual;
    else if(role==='child')snapshot.children[side].frame[branch]!.choices[offset]=actual;
    else{const parent=snapshot.parents[side];if(parent.kind==='node')parent.node.frame[source.branch]!.choices[source.step-parent.node.plan.source.step]=actual;else parent.root.frame[source.branch]!.choices[source.step-1]=actual;}
  }
  snapshot.children.forEach((node,side)=>node.plan.source.choice=parentSelection(snapshot,side,0).choice);
}
it('matches 56 independent three-version PyTorch traces, root-path ancestry, horizons and recent/cumulative changes without mutating inputs',async()=>{
  const memory=tf.memory().numTensors,{loaded,engine,predict}=runtime();
  try{
    expect(reference.modelRevision).toBe(loaded.manifest.modelRevision);expect(reference.cases).toHaveLength(56);let missingOriginal=false,differentOriginals=false;
    for(const example of reference.cases){
      const workspace=lineageProbabilityWorkspace(example.rebranchReferenceIndex,example.targetTokenId);expect(parseWorkspace(workspace,loaded.manifest)).toEqual(workspace);
      const snapshot=captureLineageStories(workspace,workspace.sharedForks!.at(-1)!),branch=example.branch as StorySelection['branch'];
      snapshot.originals.forEach((origin,side)=>{expect({branch:origin.branch,links:origin.links}).toEqual(example.origins[side]);expect(origin.links.at(-1)!.nodeId).toBe(snapshot.children[side].id);});
      differentOriginals ||=snapshot.originals[0].branch!==snapshot.originals[1].branch;
      await cpuChoices(snapshot,branch,predict);const before=structuredClone(snapshot),rows=empty();
      await traceLineageProbabilities(snapshot,branch,example.targetTokenId,loaded.manifest.config,predict,(side,role,point)=>rows[side][role].push(point),()=>false);
      const counts=lineageCounts(snapshot,branch);expect(Math.max(...counts.map(p=>p.child))).toBe(Math.max(...example.rows.map(p=>p.child.length)));
      for(const side of [0,1])for(const role of roles){
        expect(rows[side][role]).toHaveLength(example.rows[side][role].length);
        rows[side][role].forEach((point,i)=>{const expected=example.rows[side][role][i];expect([point.offset,point.source.step,point.source.choice.tokenId,point.targetTokenId]).toEqual([expected.offset,expected.step,expected.tokenId,example.targetTokenId]);expect(point.source.prefix).toEqual(expected.prefix);expect(selectionEdits(point.source)).toEqual(expected.edits);expect(point.probability).toBeCloseTo(expected.probability,5);expect(point.source.choice.probability).toBeCloseTo(expected.recordedProbability,5);});
      }
      const recent=parentDifferences(rows,example.targetTokenId),cumulative=originalDifferences(rows,example.targetTokenId,'child'),earlier=originalDifferences(rows,example.targetTokenId,'parent');
      for(const side of [0,1]){
        expect(cumulative[side]).toHaveLength(Math.min(counts[side].original,counts[side].child));
        cumulative[side].forEach(p=>{expect(p.difference).toBe(p.comparison.probability-p.original.probability);const a=recent[side].find(row=>row.offset===p.offset),b=earlier[side].find(row=>row.offset===p.offset);if(a&&b)expect(a.difference+b.difference).toBeCloseTo(p.difference,12);});
        if(example.branch==='original')expect(recent[side].every(p=>p.difference===0)).toBe(true);
        if(counts[side].original<counts[side].child){missingOriginal=true;expect(cumulative[side].some(p=>p.offset===counts[side].original)).toBe(false);}
      }
      expect(snapshot).toEqual(before);
    }
    expect(missingOriginal).toBe(true);expect(differentOriginals).toBe(true);
    // Repeated branches can outlive all saved original history; their recent comparison still works.
    const beyond=lineageProbabilityWorkspace(),root=beyond.root!;root.frame.step=10;for(const path of ['original','edited'] as const){const track=root.frame[path]!;track.choices.splice(10);track.tokenIds.splice(root.tokens.length+10);delete track.finish;}
    const late=captureLineageStories(beyond,beyond.sharedForks!.at(-1)!);expect(lineageCounts(late,'edited').map(track=>track.original)).toEqual([0,0]);await cpuChoices(late,'edited',predict);const lateRows=empty();
    await traceLineageProbabilities(late,'edited',471,loaded.manifest.config,predict,(side,role,p)=>lateRows[side][role].push(p),()=>false);
    expect(lateRows.map(track=>track.original)).toEqual([[],[]]);expect(originalDifferences(lateRows,471,'child')).toEqual([[],[]]);expect(parentDifferences(lateRows,471).map(track=>track.length)).toEqual([2,2]);
  }finally{engine.dispose();}expect(tf.memory().numTensors).toBe(memory);
},120000);
it('resolves direct root parents and rejects missing, cyclic or changed source links at any ancestry depth',()=>{
  const direct=parentProbabilityWorkspace(-1),snapshot=captureLineageStories(direct,direct.sharedForks!.at(-1)!);
  for(const side of [0,1]){expect(snapshot.originals[side].links).toHaveLength(1);expect(originalSelection(snapshot,side,0)).toEqual(parentSelection(snapshot,side,0));}
  expect(lineageCounts(snapshot,'edited').map(track=>track.original)).toEqual([2,2]);
  const asymmetric=lineageProbabilityWorkspace(),other=lineageProbabilityWorkspace(6);asymmetric.nodes[asymmetric.nodes.length-1]=structuredClone(other.nodes.at(-1)!);
  const aligned=captureLineageStories(asymmetric,asymmetric.sharedForks!.at(-1)!);expect([originalSelection(aligned,0,0).step,originalSelection(aligned,1,0).step]).toEqual([11,9]);expect(originalSelection(aligned,0,0).prefix).not.toEqual(originalSelection(aligned,1,0).prefix);
  const workspace=lineageProbabilityWorkspace(),group=workspace.sharedForks!.at(-1)!;
  const missing=structuredClone(workspace);missing.nodes=missing.nodes.filter(n=>n.id!=='trajectory-parent');expect(()=>captureLineageStories(missing,group)).toThrow('missing');
  const cycle=structuredClone(workspace);cycle.nodes[0].parentId=cycle.nodes[0].id;expect(()=>captureLineageStories(cycle,group)).toThrow('cycle');
  const changed=structuredClone(workspace);changed.nodes.find(n=>n.id==='parent-child-0')!.plan.source.choice.probability*=.5;expect(()=>captureLineageStories(changed,group)).toThrow('ancestor');
  const noRoot=structuredClone(workspace);noRoot.root=null;expect(()=>captureLineageStories(noRoot,group)).toThrow('original source story');
  const before=structuredClone(workspace),captured=captureLineageStories(workspace,group);captured.originals[0].root.tokens[0]=33;captured.originals[0].links[0].sourceStep++;expect(workspace).toEqual(before);
});
it('keeps measured zeroes but excludes missing, differently targeted or globally misaligned original comparisons',()=>{
  const source={step:12,prefix:[0,4]} as StorySelection,point={offset:0,targetTokenId:471,probability:0,source},rows=empty();
  rows[0].original=[point];rows[0].parent=[structuredClone(point)];rows[0].child=[structuredClone(point)];expect(originalDifferences(rows,471,'child')[0][0].difference).toBe(0);expect(originalDifferences(rows,471,'parent')[0][0].samePrefix).toBe(true);
  rows[0].child[0].source.step++;expect(originalDifferences(rows,471,'child')[0]).toEqual([]);expect(originalDifferences(rows,471,'parent')[0]).toHaveLength(1);
  rows[0].parent[0].targetTokenId=1;expect(originalDifferences(rows,471,'parent')[0]).toEqual([]);expect(originalDifferences(rows,1,'parent')[0]).toEqual([]);expect(originalDifferences(rows,471,'child')[1]).toEqual([]);
});
it('cancels without publishing unfinished original points and strictly checks the actual original choice for another target',async()=>{
  const {loaded,engine,predict}=runtime(),workspace=lineageProbabilityWorkspace(),snapshot=captureLineageStories(workspace,workspace.sharedForks!.at(-1)!);
  try{
    await cpuChoices(snapshot,'edited',predict);const before=structuredClone(snapshot),rows=empty();let calls=0;
    await traceLineageProbabilities(snapshot,'edited',471,loaded.manifest.config,async(ids,edits)=>{calls++;snapshot.originals[0].root.frame.original.tokenIds[0]=33;expect(ids[0]).toBe(0);return predict(ids,edits);},(side,role,point)=>rows[side][role].push(point),()=>calls>=3);
    expect(rows[0].parent).toHaveLength(1);expect(rows[0].child).toHaveLength(1);expect(rows[0].original).toHaveLength(0);expect(rows[1]).toEqual(empty()[1]);expect(originalDifferences(rows,471,'child')).toEqual([[],[]]);
    const bad=structuredClone(before),original=bad.originals[0];original.root.frame[original.branch]!.choices[bad.children[0].plan.source.step-1].probability*=.5;
    await expect(traceLineageProbabilities(bad,'edited',1,loaded.manifest.config,predict,()=>{},()=>false)).rejects.toThrow('could not be reproduced');
    for(const target of [0,2,-1,loaded.manifest.config.vocabularySize,NaN,1.5])await expect(traceLineageProbabilities(before,'edited',target,loaded.manifest.config,async()=>{throw new Error('must not infer');},()=>{},()=>false)).rejects.toThrow('eligible');
    await traceLineageProbabilities(before,'edited',471,loaded.manifest.config,async()=>{throw new Error('must not infer');},()=>{},()=>true);
  }finally{engine.dispose();}
});
