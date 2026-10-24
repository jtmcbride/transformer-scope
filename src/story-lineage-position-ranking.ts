import { requireInput } from './engine/contracts';
import { proposedEdits, selectForkToken } from './story-fork';
import { originalDifferences, originalSelection, type LineageProbabilityRows, type LineageSnapshot, type OriginalDifference } from './story-lineage-probabilities';
import type { StorySelection } from './continuation';
import type { PositionDifferenceOrder, TrajectoryBranchDraft } from './story-trajectory-differences';

export interface LineagePositionChange { offset:number; a:OriginalDifference; b:OriginalDifference }
/** Cumulative changes need original/latest values in both stories, even when their immediate parents have ended. */
export function lineagePositionChanges(rows:LineageProbabilityRows,targetTokenId:number):LineagePositionChange[]{
  const [a,b]=originalDifferences(rows,targetTokenId,'child'),peers=new Map(b.map(point=>[point.offset,point]));
  const valid=(point:OriginalDifference)=>Number.isInteger(point.offset)&&point.offset>=0&&[point.original.probability,point.comparison.probability].every(p=>Number.isFinite(p)&&p>=0&&p<=1);
  return a.flatMap(point=>{const peer=peers.get(point.offset);return peer&&valid(point)&&valid(peer)?[{offset:point.offset,a:point,b:peer}]:[];}).sort((a,b)=>a.offset-b.offset);
}
export function lineagePositionScore(row:LineagePositionChange,order:PositionDifferenceOrder):number{
  return order==='absolute'?Math.max(Math.abs(row.a.difference),Math.abs(row.b.difference))
    :order==='increase'?Math.max(row.a.difference,row.b.difference):Math.min(row.a.difference,row.b.difference);
}
export function rankLineagePositions(rows:LineagePositionChange[],order:PositionDifferenceOrder):LineagePositionChange[]{
  return [...rows].sort((a,b)=>(order==='decrease'?lineagePositionScore(a,order)-lineagePositionScore(b,order):lineagePositionScore(b,order)-lineagePositionScore(a,order))||a.offset-b.offset);
}
/** Branch from the latest actual predictions, never from historical original text or a forced target token. */
export function lineagePositionBranchDraft(snapshot:LineageSnapshot,branch:StorySelection['branch'],point:LineagePositionChange):TrajectoryBranchDraft{
  requireInput(Number.isInteger(point.offset)&&point.offset>=0,'Choose a measured generation offset.');
  const {group,children}=snapshot,pairs=children.map(node=>proposedEdits(node.plan));
  requireInput(group.nodeIds.every((id,i)=>id===children[i].id),'The ranked positions belong to another comparison.');
  requireInput(pairs[0].length===2&&JSON.stringify(pairs[0])===JSON.stringify(pairs[1]),'Both stories must use the same selected pair.');
  const target=point.a.comparison.targetTokenId;
  requireInput(Number.isInteger(target)&&target>=1&&target!==2,'Choose an eligible next token from this model.');
  const sources=children.map((node,side)=>{
    const measured=side===0?point.a:point.b,source=selectForkToken(node.plan,node.frame,branch,point.offset);
    requireInput([measured.original,measured.comparison].every(row=>row.offset===point.offset&&row.targetTokenId===target&&Number.isFinite(row.probability)&&row.probability>=0&&row.probability<=1)
      &&measured.offset===point.offset&&measured.difference===measured.comparison.probability-measured.original.probability
      &&JSON.stringify(source)===JSON.stringify(measured.comparison.source)
      &&JSON.stringify(originalSelection(snapshot,side,point.offset))===JSON.stringify(measured.original.source),'The ranked position no longer matches its original and latest story.');
    return source;
  }) as [StorySelection,StorySelection];
  return structuredClone({sources,parents:[children[0].id,children[1].id],labels:group.labels.map((label,i)=>`${label.slice(0,140)} · ${branch==='edited'?'changed':'source'} path · step ${sources[i].step}`),
    edits:pairs[0],targetTokenId:target,origin:{kind:'lineage-probability',groupId:group.id,branch,index:point.offset}}) as TrajectoryBranchDraft;
}
