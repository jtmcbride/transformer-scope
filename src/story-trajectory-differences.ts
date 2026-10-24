import type { StorySelection } from './continuation';
import { requireInput, type HeadScaleIntervention } from './engine/contracts';
import { proposedEdits, selectForkToken } from './story-fork';
import type { InteractionPoint, InteractionTracks } from './story-interaction-trajectory';
import type { SharedStoryGroup, StoryNode } from './story-workspace';

export type PositionDifferenceOrder = 'absolute' | 'increase' | 'decrease';
export interface PositionDifference { index:number; a:InteractionPoint; b:InteractionPoint; difference:number }
export interface TrajectoryBranchDraft {
  targetTokenId:number; sources:[StorySelection,StorySelection]; parents:[string,string]; labels:[string,string];
  edits:[HeadScaleIntervention,HeadScaleIntervention]; origin:{kind?:'parent-probability'|'lineage-probability';groupId:string;branch:StorySelection['branch'];index:number};
}
/** Join exact generation offsets for one fixed target. Missing or differently targeted points are never zero-filled. */
export function trajectoryDifferences(tracks:InteractionTracks,targetTokenId:number):PositionDifference[]{
  const b=new Map(tracks[1].map(point=>[point.index,point]));
  return tracks[0].flatMap(a=>{
    const peer=b.get(a.index);
    return peer&&a.targetTokenId===targetTokenId&&peer.targetTokenId===targetTokenId
      ?[{index:a.index,a,b:peer,difference:peer.delta-a.delta}]:[];
  }).sort((a,b)=>a.index-b.index);
}
export function rankTrajectoryDifferences(rows:PositionDifference[],order:PositionDifferenceOrder):PositionDifference[]{
  return [...rows].sort((a,b)=>(order==='absolute'?Math.abs(b.difference)-Math.abs(a.difference):order==='increase'?b.difference-a.difference:a.difference-b.difference)||a.index-b.index);
}
/** Branch from the actual saved paths, preserving their source edits, expiry and sampling counters. */
export function trajectoryBranchDraft(group:SharedStoryGroup,nodes:[StoryNode,StoryNode],branch:StorySelection['branch'],point:PositionDifference):TrajectoryBranchDraft{
  requireInput(group.nodeIds.every((id,i)=>id===nodes[i].id),'The ranked positions belong to another comparison.');
  const pairs=nodes.map(node=>proposedEdits(node.plan));
  requireInput(pairs[0].length===2&&JSON.stringify(pairs[0])===JSON.stringify(pairs[1]),'Both stories must use the same selected pair.');
  requireInput(point.a.targetTokenId===point.b.targetTokenId,'Choose the same target in both stories.');
  const sources=nodes.map((node,i)=>{
    const source=selectForkToken(node.plan,node.frame,branch,point.index),measured=i===0?point.a:point.b;
    requireInput(measured.index===point.index&&source.step===measured.step&&source.choice.tokenId===measured.tokenId&&source.choice.probability===measured.recordedProbability,'The ranked position no longer matches its saved story.');
    return source;
  }) as [StorySelection,StorySelection];
  return structuredClone({sources,parents:[nodes[0].id,nodes[1].id],labels:group.labels.map((label,i)=>`${label.slice(0,140)} · ${branch==='edited'?'changed':'source'} path · step ${sources[i].step}`) as [string,string],
    edits:pairs[0] as [HeadScaleIntervention,HeadScaleIntervention],targetTokenId:point.a.targetTokenId,origin:{groupId:group.id,branch,index:point.index}});
}
