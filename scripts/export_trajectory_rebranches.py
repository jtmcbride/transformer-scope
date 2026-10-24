"""Independent PyTorch strongest-offset ranking and nested two-story continuations."""
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
    reference=json.loads((ROOT/'tests/fixtures/story-fixed-token-trajectories-reference.json').read_text())['targets'][0]
    cases=[]
    with torch.no_grad():
        for case_index,group in enumerate(workspace['sharedForks']):
            nodes=[next(n for n in workspace['nodes'] if n['id']==node_id) for node_id in group['nodeIds']]
            for branch in ['original','edited']:
                tracks=reference['cases'][case_index][branch]
                gaps=[dict(index=i,difference=b['delta']-a['delta']) for i,(a,b) in enumerate(zip(*tracks))]
                ranked=sorted(gaps,key=lambda row:(-abs(row['difference']),row['index']))
                index=ranked[0]['index'];sources=[]
                for node in nodes:
                    plan,frame=node['plan'],node['frame'][branch]
                    sources.append(dict(branch=branch,step=plan['source']['step']+index,prefix=frame['tokenIds'][:len(plan['source']['prefix'])+index],choice=copy.deepcopy(frame['choices'][index]),settings=plan['source']['settings'],edits=edits_at(plan,branch=='edited',index),continuationEdits=edits_at(plan,branch=='edited',index+1)))
                for strengths in ['same','normal']:
                    pair=copy.deepcopy([nodes[0]['plan']['edit'],*nodes[0]['plan']['additionalEdits']])
                    if strengths=='normal':
                        for edit in pair:edit['factor']=1
                    for duration in ['continuation','prediction']:
                        plans=[dict(source=source,edit=pair[0],additionalEdits=[pair[1]],duration=duration,maxTokens=4) for source in sources];frames=[]
                        for plan in plans:
                            frame={}
                            for path in ['original','edited']:
                                ids=list(plan['source']['prefix']);choices=[]
                                for offset in range(4):
                                    logits,_=model(torch.tensor([ids]),edits_at(plan,path=='edited',offset),capture_activations=False)
                                    values=logits[0,-1];masked=values.clone();masked[0]=masked[2]=float('-inf');token=int(masked.argmax())
                                    choices.append(dict(tokenId=token,probability=float(values.softmax(-1)[token])));ids.append(token)
                                    if token==1 or len(ids)>=model.config.contextLength:break
                                frame[path]=dict(tokenIds=ids,choices=choices)
                            frames.append(frame)
                        cases.append(dict(groupId=group['id'],nodeIds=group['nodeIds'],caseIndex=case_index,branch=branch,targetTokenId=reference['tokenId'],ranked=ranked,index=index,strengths=strengths,duration=duration,plans=plans,frames=frames))
    (ROOT/'tests/fixtures/trajectory-rebranches-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'],cases=cases),indent=2)+'\n')
    print(f'Exported {len(cases)} strongest-offset four-path rebranches with inherited settings and both duration policies.')


if __name__=='__main__':main()
