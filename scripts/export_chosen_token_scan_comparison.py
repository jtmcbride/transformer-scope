"""Same-target paired head scans and shared continuations, independently computed in PyTorch."""
import copy
import itertools
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at
ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model, _, manifest = load_model(ROOT/'public/models/tinystories')
    workspace = json.loads((ROOT/'tests/fixtures/story-trajectory-workspace.json').read_text())['workspace']
    nodes = [next(n for n in workspace['nodes'] if n['id'] == node_id) for node_id in workspace['sharedForks'][0]['nodeIds']]
    sources = []
    for node in nodes:
        plan, frame, index = node['plan'], node['frame']['edited'], 1
        sources.append(dict(branch='edited',step=plan['source']['step']+index,prefix=frame['tokenIds'][:len(plan['source']['prefix'])+index],
                            choice=frame['choices'][index],settings=plan['source']['settings'],edits=edits_at(plan,True,index),continuationEdits=edits_at(plan,True,index+1)))
    def predict(ids, edits):
        logits, _ = model(torch.tensor([ids]),edits,capture_activations=False)
        return logits[0,-1]
    def edit(n):
        return dict(id='head-scan',kind='head.scale',layer=n//model.config.heads,head=n%model.config.heads,factor=0,tokens=dict(kind='all'))
    def measure(source,target):
        before=float(predict(source['prefix'],source['edits']).softmax(-1)[target]);singles={};rows=[]
        for a,b in itertools.combinations(range(model.config.layers*model.config.heads),2):
            pair=[edit(a),edit(b)];plan=dict(source=source,edit=pair[0],additionalEdits=[pair[1]],duration='prediction',maxTokens=1)
            values=predict(source['prefix'],edits_at(plan,True,0));order=sorted((i for i in range(len(values)) if i not in (0,2)),key=lambda i:(-float(values[i]),i))
            for n in (a,b):
                if n not in singles:
                    alone=dict(source=source,edit=edit(n),duration='prediction',maxTokens=1)
                    singles[n]=float(predict(source['prefix'],edits_at(alone,True,0)).softmax(-1)[target])
            after=float(values.softmax(-1)[target]);individual=[singles[a],singles[b]];summed=sum(p-before for p in individual)
            rows.append(dict(edits=pair,before=before,after=after,delta=after-before,rank=order.index(target)+1,choice=order[0],individualProbabilities=individual,sumDelta=summed,interaction=after-before-summed))
        return rows
    def rollout(pair,duration):
        plans=[dict(source=copy.deepcopy(source),edit=pair[0],additionalEdits=[pair[1]],duration=duration,maxTokens=4) for source in sources];frames=[]
        for plan in plans:
            frame={}
            for branch in ['original','edited']:
                ids=list(plan['source']['prefix']);choices=[]
                for i in range(4):
                    values=predict(ids,edits_at(plan,branch=='edited',i));masked=values.clone();masked[0]=masked[2]=float('-inf');token=int(masked.argmax())
                    choices.append(dict(tokenId=token,probability=float(values.softmax(-1)[token])));ids.append(token)
                    if token==1 or len(ids)>=model.config.contextLength:break
                frame[branch]=dict(tokenIds=ids,choices=choices)
            frames.append(frame)
        return dict(plans=plans,frames=frames)
    cases=[]
    with torch.no_grad():
        for target in [471,1]:
            rows=[measure(source,target) for source in sources]
            changes=[dict(edits=a['edits'],difference=b['delta']-a['delta']) for a,b in zip(*rows)]
            branches={}
            for order in ['absolute','increase','decrease']:
                best=max(changes,key=lambda r:abs(r['difference']) if order=='absolute' else r['difference'] if order=='increase' else -r['difference'])
                branches[order]={duration:rollout(best['edits'],duration) for duration in ['continuation','prediction']}
            cases.append(dict(targetTokenId=target,rows=rows,changes=changes,branches=branches))
    data=dict(modelRevision=manifest['modelRevision'],nodeIds=[n['id'] for n in nodes],index=1,sources=sources,cases=cases)
    (ROOT/'tests/fixtures/chosen-token-scan-comparison-reference.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Exported 480 same-target pair measurements and 12 independent four-path continuation comparisons.')


if __name__=='__main__':main()
