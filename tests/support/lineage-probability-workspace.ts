import type { ForkPlan } from '../../src/story-fork';
import type { StoryBranch, StoryFrame } from '../../src/continuation';
import type { StoryNode, StoryWorkspace } from '../../src/story-workspace';
import { parentProbabilityWorkspace } from './parent-probability-workspace';
import rebranches from '../fixtures/parent-position-rebranches-reference.json' with {type:'json'};

export function lineageProbabilityWorkspace(rebranchReferenceIndex=2,targetTokenId=471):StoryWorkspace{
  const example=rebranches.cases[rebranchReferenceIndex],workspace=parentProbabilityWorkspace(example.referenceIndex,targetTokenId);
  const nodes=example.plans.map((p,side)=>{
    const plan=structuredClone(p) as ForkPlan;
    const path=(branch:typeof example.frames[number]['original']):StoryBranch=>({...structuredClone(branch),finish:branch.choices.at(-1)?.tokenId===1?'end of story':branch.tokenIds.length>=64?'context limit':'token limit'});
    const frame:StoryFrame={step:Math.max(example.frames[side].original.choices.length,example.frames[side].edited.choices.length),original:path(example.frames[side].original),edited:path(example.frames[side].edited)};
    return {id:`lineage-latest-${side}`,parentId:`parent-child-${side}`,plan,frame} as StoryNode;
  });
  workspace.nodes.push(...nodes);workspace.sharedForks!.push({id:'lineage-comparison',nodeIds:[nodes[0].id,nodes[1].id],labels:['Latest A','Latest B'],targetTokenId});return workspace;
}
