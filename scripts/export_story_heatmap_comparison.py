"""Independent same-step interactions after two story paths have diverged."""
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
    parent = json.loads((ROOT/'tests/fixtures/story-forks-reference.json').read_text())['cases'][0]
    plan, index = parent['plan'], 1
    cases = []
    with torch.no_grad():
        for branch in ['original', 'edited']:
            source = dict(branch=branch, step=plan['source']['step']+index,
                          prefix=parent[branch]['tokenIds'][:len(plan['source']['prefix'])+index],
                          choice=parent[branch]['choices'][index], settings=plan['source']['settings'],
                          edits=edits_at(plan, branch == 'edited', index),
                          continuationEdits=edits_at(plan, branch == 'edited', index+1))
            def probability(edits):
                proposal = dict(source=source, edit=edits[0], additionalEdits=edits[1:], duration='prediction', maxTokens=1)
                logits, _ = model(torch.tensor([source['prefix']]), edits_at(proposal, True, 0), capture_activations=False)
                return float(logits[0,-1].softmax(-1)[source['choice']['tokenId']])
            def edit(head):
                return dict(id='head-scan',kind='head.scale',layer=head[0],head=head[1],factor=0,tokens=dict(kind='all'))
            heads = list(itertools.product(range(model.config.layers),range(model.config.heads)))
            singles = {head: probability([edit(head)]) for head in heads}
            before = source['choice']['probability']
            rows = []
            for a,b in itertools.combinations(heads,2):
                after = probability([edit(a),edit(b)])
                summed = (singles[a]-before)+(singles[b]-before)
                rows.append(dict(edits=[edit(a),edit(b)],after=after,individualProbabilities=[singles[a],singles[b]],sumDelta=summed,interaction=(after-before)-summed))
            cases.append(dict(source=source,rows=rows))
    differences = [dict(edits=a['edits'],difference=b['interaction']-a['interaction']) for a,b in zip(cases[0]['rows'],cases[1]['rows'])]
    result = dict(modelRevision=manifest['modelRevision'],cases=cases,differences=differences)
    (ROOT/'tests/fixtures/story-heatmap-comparison-reference.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Exported two full 120-pair maps at global step seven and their interaction differences.')


if __name__ == '__main__':
    main()
