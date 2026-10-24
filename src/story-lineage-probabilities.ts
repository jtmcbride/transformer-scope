import { chooseToken, drawAt, selectStoryToken, selectionEdits, StoryContinuation, type StorySelection } from './continuation';
import { logSoftmax } from './engine/comparison';
import { requireInput, validateTokens, type ModelConfig } from './engine/contracts';
import { StoryFork, selectForkToken, type ForkPredict } from './story-fork';
import { captureParentStories, parentCounts, parentSelection, type ParentProbabilityPoint, type ParentProbabilityRows, type ParentSnapshot } from './story-parent-probabilities';
import type { SharedStoryGroup, StoryRoot, StoryWorkspace } from './story-workspace';

export interface AncestryLink { nodeId:string;parentId:string|null;sourceBranch:StorySelection['branch'];sourceStep:number }
export interface OriginalStory { root:StoryRoot;branch:StorySelection['branch'];links:AncestryLink[] }
export interface LineageSnapshot extends ParentSnapshot { originals:[OriginalStory,OriginalStory] }
export type LineageRole = 'original'|'parent'|'child';
export type LineageProbabilityRows = [ParentProbabilityRows[0]&{original:ParentProbabilityPoint[]},ParentProbabilityRows[1]&{original:ParentProbabilityPoint[]}];
export interface OriginalDifference { offset:number;original:ParentProbabilityPoint;comparison:ParentProbabilityPoint;difference:number;samePrefix:boolean }

/** Resolve the saved root path through real source links, independently for A and B. */
export function captureLineageStories(workspace:StoryWorkspace,group:SharedStoryGroup):LineageSnapshot{
  const snapshot=captureParentStories(workspace,group);
  requireInput(!!workspace.root,'The original source story is missing.');
  const originals=snapshot.children.map(child=>{
    let node=child;const seen=new Set<string>(),links:AncestryLink[]=[];
    while(true){
      requireInput(!seen.has(node.id),'The story ancestry contains a cycle.');seen.add(node.id);
      const source=node.plan.source;links.unshift({nodeId:node.id,parentId:node.parentId,sourceBranch:source.branch,sourceStep:source.step});
      const parent=node.parentId===null?undefined:workspace.nodes.find(item=>item.id===node.parentId);
      requireInput(node.parentId===null||!!parent,'An original-story ancestor is missing.');
      requireInput(!parent||!seen.has(parent.id),'The story ancestry contains a cycle.');
      const expected=parent?selectForkToken(parent.plan,parent.frame,source.branch,source.step-parent.plan.source.step)
        :selectStoryToken(workspace.root!.frame,workspace.root!.tokens.length,source.branch,source.step-1,workspace.root!.settings,workspace.root!.edit);
      requireInput(JSON.stringify(expected)===JSON.stringify(source),'A story no longer matches its actual ancestor.');
      if(!parent)return {root:workspace.root!,branch:source.branch,links};
      node=parent;
    }
  }) as [OriginalStory,OriginalStory];
  return structuredClone({...snapshot,originals});
}
export function originalSelection(snapshot:LineageSnapshot,side:number,offset:number):StorySelection{
  const {root,branch}=snapshot.originals[side],step=snapshot.children[side].plan.source.step+offset;
  return selectStoryToken(root.frame,root.tokens.length,branch,step-1,root.settings,root.edit);
}
/** Compare only the generation horizon of the latest two continuations; older missing tails stay unavailable. */
export function lineageCounts(snapshot:LineageSnapshot,branch:StorySelection['branch']):[{original:number;parent:number;child:number},{original:number;parent:number;child:number}]{
  const counts=parentCounts(snapshot,branch),horizon=Math.max(...counts.map(track=>track.child));
  return counts.map((track,side)=>({child:track.child,parent:Math.min(track.parent,horizon),original:Math.min(horizon,Math.max(0,snapshot.originals[side].root.frame[snapshot.originals[side].branch]!.choices.length-(snapshot.children[side].plan.source.step-1)))})) as ReturnType<typeof lineageCounts>;
}
/** The historical original is joined only at the same global step and fixed target. No missing value becomes zero. */
export function originalDifferences(rows:LineageProbabilityRows,targetTokenId:number,role:'parent'|'child'):[OriginalDifference[],OriginalDifference[]]{
  return rows.map(track=>{
    const peers=new Map(track[role].map(point=>[point.offset,point]));
    return track.original.flatMap(original=>{const comparison=peers.get(original.offset);return comparison&&original.targetTokenId===targetTokenId&&comparison.targetTokenId===targetTokenId&&original.source.step===comparison.source.step
      ?[{offset:original.offset,original,comparison,difference:comparison.probability-original.probability,samePrefix:JSON.stringify(original.source.prefix)===JSON.stringify(comparison.source.prefix)}]:[];});
  }) as ReturnType<typeof originalDifferences>;
}
export async function traceLineageProbabilities(snapshot:LineageSnapshot,branch:StorySelection['branch'],targetTokenId:number,config:ModelConfig,predict:ForkPredict,
  emit:(side:0|1,role:LineageRole,point:ParentProbabilityPoint)=>void,cancelled:()=>boolean):Promise<void>{
  requireInput(Number.isInteger(targetTokenId)&&targetTokenId>=1&&targetTokenId!==2&&targetTokenId<config.vocabularySize,'Choose an eligible next token from this model.');
  const captured=structuredClone(snapshot),counts=lineageCounts(captured,branch);
  for(const node of captured.children)new StoryFork(node.plan,config).restore(node.frame);
  for(const parent of captured.parents)if(parent.kind==='node')new StoryFork(parent.node.plan,config).restore(parent.node.frame);
  for(const original of captured.originals)new StoryContinuation(original.root.tokens,original.root.settings,config.contextLength,original.root.edit).restore(original.root.frame);
  const count=Math.max(...counts.flatMap(track=>[track.original,track.parent,track.child]));
  for(let offset=0;offset<count;offset++){
    const cache=new Map<string,ArrayLike<number>>();
    for(const side of [0,1] as const)for(const role of ['parent','child','original'] as const){
      if(cancelled())return;if(offset>=counts[side][role])continue;
      const source=role==='original'?originalSelection(captured,side,offset):role==='parent'?parentSelection(captured,side,offset):selectForkToken(captured.children[side].plan,captured.children[side].frame,branch,offset);
      validateTokens(source.prefix,config);const edits=selectionEdits(source),key=JSON.stringify([source.prefix,edits]);
      if(!cache.has(key))cache.set(key,await predict([...source.prefix],edits));
      if(cancelled())return;const values=cache.get(key)!,replay=chooseToken(values,source.settings,drawAt(source.settings.seed,source.step-1));
      requireInput(replay.tokenId===source.choice.tokenId&&Math.abs(replay.probability-source.choice.probability)<=1e-6,'This original, parent or latest token could not be reproduced.');
      emit(side,role,{offset,targetTokenId,probability:Math.exp(logSoftmax(values)[targetTokenId]),source});
    }
  }
}
