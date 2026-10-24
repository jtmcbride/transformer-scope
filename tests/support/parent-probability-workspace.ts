import type { StoryWorkspace, StoryNode } from '../../src/story-workspace';
import type { ForkPlan } from '../../src/story-fork';
import type { StoryBranch, StoryFrame } from '../../src/continuation';
import input from '../fixtures/story-trajectory-workspace.json' with {type:'json'};
import branches from '../fixtures/trajectory-rebranches-reference.json' with {type:'json'};
import reference from '../fixtures/parent-probabilities-reference.json' with {type:'json'};

export function parentProbabilityWorkspace(referenceIndex=6,targetTokenId=471):StoryWorkspace{
  const workspace=structuredClone(input.workspace) as unknown as StoryWorkspace;
  const example=referenceIndex===-1?reference.root:branches.cases[referenceIndex];
  const nodes=example.plans.map((p,side)=>{
    const plan=structuredClone(p) as ForkPlan;
    const path=(branch:typeof example.frames[number]['original']):StoryBranch=>({...structuredClone(branch),finish:branch.choices.at(-1)?.tokenId===1?'end of story':branch.tokenIds.length>=64?'context limit':'token limit'});
    const frame:StoryFrame={step:Math.max(example.frames[side].original.choices.length,example.frames[side].edited.choices.length),original:path(example.frames[side].original),edited:path(example.frames[side].edited)};
    return {id:`parent-child-${side}`,parentId:referenceIndex===-1?null:branches.cases[referenceIndex].nodeIds[side],plan,frame} as StoryNode;
  });
  workspace.nodes.push(...nodes);workspace.sharedForks!.push({id:'parent-comparison',nodeIds:[nodes[0].id,nodes[1].id],labels:['Rebranched A','Rebranched B'],targetTokenId});return workspace;
}
