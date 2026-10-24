import { chooseToken, drawAt, selectStoryToken, selectionEdits, type StorySelection } from './continuation';
import { logSoftmax } from './engine/comparison';
import { requireInput, validateTokens, type ModelConfig } from './engine/contracts';
import { StoryFork, selectForkToken, type ForkPredict } from './story-fork';
import type { SharedStoryGroup, StoryNode, StoryRoot, StoryWorkspace } from './story-workspace';

export type ParentRecord = {kind:'node';node:StoryNode} | {kind:'root';root:StoryRoot};
export interface ParentSnapshot { group:SharedStoryGroup; children:[StoryNode,StoryNode]; parents:[ParentRecord,ParentRecord] }
export type ParentRole = 'parent' | 'child';
export interface ParentProbabilityPoint { offset:number; targetTokenId:number; probability:number; source:StorySelection }
export type ParentProbabilityRows = [{parent:ParentProbabilityPoint[];child:ParentProbabilityPoint[]},{parent:ParentProbabilityPoint[];child:ParentProbabilityPoint[]}];
export interface ParentProbabilityDifference { offset:number;parent:ParentProbabilityPoint;child:ParentProbabilityPoint;difference:number;samePrefix:boolean }
export function captureParentStories(workspace:StoryWorkspace,group:SharedStoryGroup):ParentSnapshot{
  const children=group.nodeIds.map(id=>workspace.nodes.find(node=>node.id===id));
  requireInput(children.every(Boolean),'The saved branches are missing.');
  const parents=children.map(child=>{
    if(child!.parentId===null){requireInput(!!workspace.root,'The source story is missing.');return {kind:'root',root:workspace.root!} as ParentRecord;}
    const node=workspace.nodes.find(node=>node.id===child!.parentId);requireInput(!!node,'The parent comparison is missing.');return {kind:'node',node:node!} as ParentRecord;
  });
  const captured=structuredClone({group,children,parents}) as ParentSnapshot;
  captured.children.forEach((child,side)=>requireInput(JSON.stringify(parentSelection(captured,side,0))===JSON.stringify(child.plan.source),'A branch no longer matches its actual parent.'));
  return captured;
}
export function parentSelection(snapshot:ParentSnapshot,side:number,offset:number):StorySelection{
  const {source}=snapshot.children[side].plan,parent=snapshot.parents[side];
  return parent.kind==='node'?selectForkToken(parent.node.plan,parent.node.frame,source.branch,source.step-parent.node.plan.source.step+offset)
    :selectStoryToken(parent.root.frame,parent.root.tokens.length,source.branch,source.step-1+offset,parent.root.settings,parent.root.edit);
}
export function parentCounts(snapshot:ParentSnapshot,childBranch:StorySelection['branch']):[{parent:number;child:number},{parent:number;child:number}]{
  return snapshot.children.map((child,side)=>{
    const parent=snapshot.parents[side],source=child.plan.source;
    const count=parent.kind==='node'?parent.node.frame[source.branch]!.choices.length-(source.step-parent.node.plan.source.step)
      :parent.root.frame[source.branch]!.choices.length-(source.step-1);
    return {parent:count,child:child.frame[childBranch]!.choices.length};
  }) as ReturnType<typeof parentCounts>;
}
export function parentDifferences(rows:ParentProbabilityRows,targetTokenId:number):[ParentProbabilityDifference[],ParentProbabilityDifference[]]{
  return rows.map(track=>{
    const peers=new Map(track.child.map(point=>[point.offset,point]));
    return track.parent.flatMap(parent=>{const child=peers.get(parent.offset);return child&&child.targetTokenId===targetTokenId&&parent.targetTokenId===targetTokenId&&child.source.step===parent.source.step
      ?[{offset:parent.offset,parent,child,difference:child.probability-parent.probability,samePrefix:JSON.stringify(child.source.prefix)===JSON.stringify(parent.source.prefix)}]:[];});
  }) as ReturnType<typeof parentDifferences>;
}
/** Replay each actual saved prediction, then read the fixed target on that same path's own prefix and active edits. */
export async function traceParentProbabilities(snapshot:ParentSnapshot,childBranch:StorySelection['branch'],targetTokenId:number,config:ModelConfig,predict:ForkPredict,
  emit:(side:0|1,role:ParentRole,point:ParentProbabilityPoint)=>void,cancelled:()=>boolean):Promise<void>{
  requireInput(Number.isInteger(targetTokenId)&&targetTokenId>=1&&targetTokenId!==2&&targetTokenId<config.vocabularySize,'Choose an eligible next token from this model.');
  const captured=structuredClone(snapshot),counts=parentCounts(captured,childBranch);
  for(const node of captured.children)new StoryFork(node.plan,config).restore(node.frame);
  for(const parent of captured.parents)if(parent.kind==='node')new StoryFork(parent.node.plan,config).restore(parent.node.frame);
  const count=Math.max(...counts.flatMap(track=>[track.parent,track.child]));
  for(let offset=0;offset<count;offset++){
    // Equal prefixes/settings at the fork or on an unchanged path share one distribution; retain no model runs or tensors.
    const cache=new Map<string,ArrayLike<number>>();
    for(const side of [0,1] as const)for(const role of ['parent','child'] as const){
      if(cancelled())return;if(offset>=counts[side][role])continue;
      const source=role==='parent'?parentSelection(captured,side,offset):selectForkToken(captured.children[side].plan,captured.children[side].frame,childBranch,offset);
      validateTokens(source.prefix,config);
      const edits=selectionEdits(source),key=JSON.stringify([source.prefix,edits]);
      if(!cache.has(key))cache.set(key,await predict([...source.prefix],edits));
      if(cancelled())return;const values=cache.get(key)!;
      const replay=chooseToken(values,source.settings,drawAt(source.settings.seed,source.step-1));
      requireInput(replay.tokenId===source.choice.tokenId&&Math.abs(replay.probability-source.choice.probability)<=1e-6,'This parent or branch token could not be reproduced.');
      const probability=Math.exp(logSoftmax(values)[targetTokenId]);
      emit(side,role,{offset,targetTokenId,probability,source});
    }
  }
}
