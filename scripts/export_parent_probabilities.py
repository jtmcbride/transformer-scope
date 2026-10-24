"""Independent actual-parent versus saved-child next-token probabilities (PyTorch)."""
import copy
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at
ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model,_,manifest=load_model(ROOT/'public/models/tinystories')
    workspace=json.loads((ROOT/'tests/fixtures/story-trajectory-workspace.json').read_text())['workspace']
    branches=json.loads((ROOT/'tests/fixtures/trajectory-rebranches-reference.json').read_text())['cases']
    cache={}
    def predict(ids,edits):
        key=json.dumps([ids,edits],sort_keys=True)
        if key not in cache:
            logits,_=model(torch.tensor([ids]),edits,capture_activations=False);cache[key]=logits[0,-1].softmax(-1)
        return cache[key]
    root=workspace['root'];root_plans=[];root_frames=[];root_index=14
    with torch.no_grad():
        for path in ['original','edited']:
            source=dict(branch=path,step=root_index+1,prefix=root['frame'][path]['tokenIds'][:len(root['tokens'])+root_index],choice=root['frame'][path]['choices'][root_index],settings=root['settings'])
            if path=='edited':source['edit']=root['edit']
            pair=copy.deepcopy([branches[0]['plans'][0]['edit'],*branches[0]['plans'][0]['additionalEdits']])
            plan=dict(source=source,edit=pair[0],additionalEdits=[pair[1]],duration='prediction',maxTokens=4);frame={}
            for branch in ['original','edited']:
                ids=list(source['prefix']);choices=[]
                for index in range(4):
                    probabilities=predict(ids,edits_at(plan,branch=='edited',index));eligible=probabilities.clone();eligible[0]=eligible[2]=-1;token=int(eligible.argmax())
                    choices.append(dict(tokenId=token,probability=float(probabilities[token])));ids.append(token)
                    if token==1 or len(ids)>=model.config.contextLength:break
                frame[branch]=dict(tokenIds=ids,choices=choices)
            root_plans.append(plan);root_frames.append(frame)
        examples=[dict(kind='nested',referenceIndex=i,plans=row['plans'],frames=row['frames'],nodeIds=row['nodeIds']) for i,row in enumerate(branches)]
        examples.append(dict(kind='root',referenceIndex=-1,plans=root_plans,frames=root_frames,nodeIds=[None,None]))
        cases=[]
        for example in examples:
            for child_branch in ['original','edited']:
                for target in [471,1]:
                    tracks=[]
                    for side,plan in enumerate(example['plans']):
                        source=plan['source'];parent=next((n for n in workspace['nodes'] if n['id']==example['nodeIds'][side]),None)
                        parent_frame=parent['frame'][source['branch']] if parent else root['frame'][source['branch']]
                        start=source['step']-parent['plan']['source']['step'] if parent else source['step']-1
                        measured={}
                        for role in ['parent','child']:
                            frame=parent_frame if role=='parent' else example['frames'][side][child_branch]
                            first=start if role=='parent' else 0;rows=[]
                            for index in range(first,len(frame['choices'])):
                                offset=index-first;prefix=frame['tokenIds'][:len(source['prefix'])+offset]
                                if role=='child':edits=edits_at(plan,child_branch=='edited',index)
                                elif parent:edits=edits_at(parent['plan'],source['branch']=='edited',index)
                                else:edits=[root['edit']] if source['branch']=='edited' else []
                                probability=float(predict(prefix,edits)[target]);choice=frame['choices'][index]
                                rows.append(dict(offset=offset,step=source['step']+offset,tokenId=choice['tokenId'],recordedProbability=choice['probability'],targetTokenId=target,probability=probability,prefix=prefix,edits=edits))
                            measured[role]=rows
                        tracks.append(measured)
                    cases.append(dict(kind=example['kind'],referenceIndex=example['referenceIndex'],childBranch=child_branch,targetTokenId=target,rows=tracks))
    data=dict(modelRevision=manifest['modelRevision'],root=dict(plans=root_plans,frames=root_frames),cases=cases)
    (ROOT/'tests/fixtures/parent-probabilities-reference.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f'Exported {len(cases)} independent parent/child traces; {len(cache)} actual distributions shared across word/EOS measurements.')


if __name__=='__main__':main()
