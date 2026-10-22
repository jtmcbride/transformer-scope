"""Independent full head scans on original and inherited-edit token contexts."""
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
model, _, manifest = load_model(ROOT / 'public/models/tinystories')
a = json.loads((ROOT/'tests/fixtures/story-forks-reference.json').read_text())['cases'][0]['plan']['source']
b = json.loads((ROOT/'tests/fixtures/nested-story-forks-reference.json').read_text())['cases'][2]['plan']['source']
cases=[]
with torch.no_grad():
    for source in [a,b]:
        rows=[]
        for layer in range(model.config.layers):
            for head in range(model.config.heads):
                for factor in [0,1,2]:
                    edit=dict(id='head-scan',kind='head.scale',layer=layer,head=head,factor=factor,tokens=dict(kind='all'))
                    plan=dict(source=source, edit=edit, duration='prediction', maxTokens=1)
                    logits,_=model(torch.tensor([source['prefix']]),edits_at(plan,True,0),capture_activations=False)
                    values=logits[0,-1]
                    order=sorted([i for i in range(len(values)) if i not in (0,2)],key=lambda i:(-float(values[i]),i))
                    prob=float(values.softmax(-1)[source['choice']['tokenId']])
                    rows.append(dict(edit=edit,after=prob,rank=order.index(source['choice']['tokenId'])+1,choice=order[0]))
        cases.append(dict(source=source,rows=rows))
(ROOT/'tests/fixtures/story-head-scan-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'],cases=cases),indent=2)+'\n')
print('Exported two full 48-change head scans.')
