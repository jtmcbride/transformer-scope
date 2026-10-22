"""Independent PyTorch references for branching again, including inherited edit expiry."""
import copy
import json
from pathlib import Path
import torch
from text_probes import load_model

ROOT = Path(__file__).resolve().parents[1]


def edits_at(plan, changed, index):
    source = plan['source']
    inherited = source.get('continuationEdits') if index else source.get('edits')
    if inherited is None:
        inherited = source.get('edits', [source['edit']] if 'edit' in source else [])
    if not changed or plan['duration'] == 'prediction' and index > 0:
        return copy.deepcopy(inherited)
    edits = [plan['edit']] + plan.get('additionalEdits', [])
    keys = {(edit['layer'], edit['head']) for edit in edits}
    return copy.deepcopy([item for item in inherited if (item['layer'], item['head']) not in keys] + edits)


def main():
    torch.set_num_threads(4)
    model, tokenizer, manifest = load_model(ROOT / 'public/models/tinystories')
    roots = json.loads((ROOT / 'tests/fixtures/story-forks-reference.json').read_text())['cases']
    cases = []
    # source collection, case, branch, local index, new layer/head, duration
    specs = [('root', 0, 'edited', 1, 1, 1, 'continuation'),
             ('root', 1, 'edited', 0, 1, 1, 'prediction'),
             ('nested', 0, 'edited', 2, 2, 2, 'continuation')]
    for collection, case_index, branch, index, layer, head, duration in specs:
        parent = (roots if collection == 'root' else cases)[case_index]
        parent_plan = parent['plan']
        selected = dict(branch=branch, step=parent_plan['source']['step'] + index,
                        prefix=parent[branch]['tokenIds'][:len(parent_plan['source']['prefix']) + index],
                        choice=parent[branch]['choices'][index], settings=parent_plan['source']['settings'],
                        edits=edits_at(parent_plan, branch == 'edited', index),
                        continuationEdits=edits_at(parent_plan, branch == 'edited', index + 1))
        depth = 1 if collection == 'root' else 2
        edit = dict(id=f'fork-{depth}-head', kind='head.scale', layer=layer, head=head, factor=0, tokens=dict(kind='all'))
        plan = dict(source=selected, edit=edit, duration=duration, maxTokens=6)
        result = dict(parent=dict(collection=collection, case=case_index, branch=branch, index=index), plan=plan)
        with torch.no_grad():
            for name in ('original', 'edited'):
                ids, choices = list(selected['prefix']), []
                for i in range(plan['maxTokens']):
                    logits, _ = model(torch.tensor([ids]), edits_at(plan, name == 'edited', i), capture_activations=False)
                    values = logits[0, -1]
                    masked = values.clone()
                    masked[0] = masked[2] = float('-inf')
                    token = int(masked.argmax())
                    choices.append(dict(tokenId=token, probability=float(values.softmax(-1)[token])))
                    ids.append(token)
                    if token == 1 or len(ids) >= model.config.contextLength:
                        break
                result[name] = dict(tokenIds=ids, choices=choices)
                print(len(cases), name, tokenizer.decode(ids))
        cases.append(result)
    (ROOT / 'tests/fixtures/nested-story-forks-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2) + '\n')


if __name__ == '__main__':
    main()
