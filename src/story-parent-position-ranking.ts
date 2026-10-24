import { requireInput } from './engine/contracts';
import { proposedEdits, selectForkToken } from './story-fork';
import { parentDifferences, parentSelection, type ParentProbabilityDifference, type ParentProbabilityRows, type ParentSnapshot } from './story-parent-probabilities';
import type { StorySelection } from './continuation';
import type { PositionDifferenceOrder, TrajectoryBranchDraft } from './story-trajectory-differences';

export interface ParentPositionChange { offset:number; a:ParentProbabilityDifference; b:ParentProbabilityDifference }
/** A rankable position has an actual parent and child measurement in both stories, for the same fixed target. */
export function parentPositionChanges(rows:ParentProbabilityRows,targetTokenId:number):ParentPositionChange[]{
  const [a,b]=parentDifferences(rows,targetTokenId),peers=new Map(b.map(point=>[point.offset,point]));
  const valid=(point:ParentProbabilityDifference)=>[point.parent.probability,point.child.probability].every(p=>Number.isFinite(p)&&p>=0&&p<=1);
  return a.flatMap(point=>{const peer=peers.get(point.offset);return peer&&valid(point)&&valid(peer)?[{offset:point.offset,a:point,b:peer}]:[];}).sort((a,b)=>a.offset-b.offset);
}
export function parentPositionScore(row:ParentPositionChange,order:PositionDifferenceOrder):number{
  return order==='absolute'?Math.max(Math.abs(row.a.difference),Math.abs(row.b.difference))
    :order==='increase'?Math.max(row.a.difference,row.b.difference):Math.min(row.a.difference,row.b.difference);
}
export function rankParentPositions(rows:ParentPositionChange[],order:PositionDifferenceOrder):ParentPositionChange[]{
  return [...rows].sort((a,b)=>(order==='decrease'?parentPositionScore(a,order)-parentPositionScore(b,order):parentPositionScore(b,order)-parentPositionScore(a,order))||a.offset-b.offset);
}
/** Branch from each new continuation's actual pre-token selection, never the historical parent or a predicted alternative. */
export function parentPositionBranchDraft(snapshot:ParentSnapshot,branch:StorySelection['branch'],point:ParentPositionChange):TrajectoryBranchDraft{
  requireInput(Number.isInteger(point.offset)&&point.offset>=0,'Choose a measured generation offset.');
  const {group,children}=snapshot,pairs=children.map(node=>proposedEdits(node.plan));
  requireInput(group.nodeIds.every((id,i)=>id===children[i].id),'The ranked positions belong to another comparison.');
  requireInput(pairs[0].length===2&&JSON.stringify(pairs[0])===JSON.stringify(pairs[1]),'Both stories must use the same selected pair.');
  const target=point.a.child.targetTokenId;
  requireInput(Number.isInteger(target)&&target>=1&&target!==2,'Choose an eligible next token from this model.');
  const sources=children.map((node,side)=>{
    const measured=side===0?point.a:point.b,source=selectForkToken(node.plan,node.frame,branch,point.offset);
    requireInput([measured.parent,measured.child].every(row=>row.offset===point.offset&&row.targetTokenId===target&&Number.isFinite(row.probability)&&row.probability>=0&&row.probability<=1)
      &&measured.offset===point.offset&&measured.difference===measured.child.probability-measured.parent.probability
      &&JSON.stringify(source)===JSON.stringify(measured.child.source)
      &&JSON.stringify(parentSelection(snapshot,side,point.offset))===JSON.stringify(measured.parent.source),'The ranked position no longer matches its saved parent and new story.');
    return source;
  }) as [StorySelection,StorySelection];
  return structuredClone({sources,parents:[children[0].id,children[1].id],labels:group.labels.map((label,i)=>`${label.slice(0,140)} · ${branch==='edited'?'changed':'source'} path · step ${sources[i].step}`),
    edits:pairs[0],targetTokenId:target,origin:{kind:'parent-probability',groupId:group.id,branch,index:point.offset}}) as TrajectoryBranchDraft;
}
