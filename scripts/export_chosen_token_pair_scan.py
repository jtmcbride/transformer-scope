"""Independent chosen-target pair scans on two real, inherited story prefixes."""
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
    cases = []
    with torch.no_grad():
        for side, target in [(0, 471), (1, 1)]:
            node = next(n for n in workspace['nodes'] if n['id'] == workspace['sharedForks'][0]['nodeIds'][side])
            parent, frame, index = node['plan'], node['frame']['edited'], 1
            source = dict(branch='edited', step=parent['source']['step']+index,
                          prefix=frame['tokenIds'][:len(parent['source']['prefix'])+index], choice=frame['choices'][index],
                          settings=parent['source']['settings'], edits=edits_at(parent, True, index),
                          continuationEdits=edits_at(parent, True, index+1))
            def predict(edits, ids=None):
                logits, _ = model(torch.tensor([source['prefix'] if ids is None else ids]), edits, capture_activations=False)
                return logits[0, -1]
            def edit(n):
                return dict(id='head-scan', kind='head.scale', layer=n//model.config.heads, head=n%model.config.heads, factor=0, tokens=dict(kind='all'))
            before = float(predict(source['edits']).softmax(-1)[target])
            singles, rows = {}, []
            for a,b in itertools.combinations(range(model.config.layers*model.config.heads), 2):
                pair = [edit(a), edit(b)]
                plan = dict(source=source, edit=pair[0], additionalEdits=[pair[1]], duration='continuation', maxTokens=4)
                values = predict(edits_at(plan, True, 0))
                order = sorted((i for i in range(len(values)) if i not in (0,2)), key=lambda i:(-float(values[i]),i))
                for n in (a,b):
                    if n not in singles:
                        alone = dict(source=source, edit=edit(n), duration='prediction', maxTokens=1)
                        singles[n] = float(predict(edits_at(alone, True, 0)).softmax(-1)[target])
                after = float(values.softmax(-1)[target]); individual = [singles[a],singles[b]]
                summed = sum(p-before for p in individual)
                rows.append(dict(edits=pair,before=before,after=after,delta=after-before,rank=order.index(target)+1,choice=order[0],individualProbabilities=individual,sumDelta=summed,interaction=after-before-summed))
            def rollout(row):
                plan = dict(source=copy.deepcopy(source), edit=row['edits'][0], additionalEdits=[row['edits'][1]], duration='continuation', maxTokens=4)
                result = dict(plan=plan)
                for branch in ['original','edited']:
                    ids, choices = list(source['prefix']), []
                    for i in range(4):
                        values = predict(edits_at(plan, branch=='edited', i), ids)
                        masked = values.clone(); masked[0] = masked[2] = float('-inf')
                        token = int(masked.argmax()); choices.append(dict(tokenId=token,probability=float(values.softmax(-1)[token]))); ids.append(token)
                        if token == 1 or len(ids) >= model.config.contextLength: break
                    result[branch] = dict(tokenIds=ids, choices=choices)
                return result
            cases.append(dict(nodeId=node['id'],index=index,source=source,targetTokenId=target,rows=rows,
                              increase=rollout(max(rows,key=lambda row:row['delta'])),decrease=rollout(min(rows,key=lambda row:row['delta']))))
    (ROOT/'tests/fixtures/chosen-token-pair-scan-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'],cases=cases),indent=2)+'\n')
    print(f'Exported {sum(len(case["rows"]) for case in cases)} independent chosen-target pair measurements and four continuations per target.')


if __name__ == '__main__': main()
