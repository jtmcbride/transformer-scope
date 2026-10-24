"""Independent original/root, immediate parent and latest-path probabilities after repeated rebranching."""
import copy
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model, _, manifest = load_model(ROOT / 'public/models/tinystories')
    initial = json.loads((ROOT / 'tests/fixtures/story-trajectory-workspace.json').read_text())['workspace']
    previous = json.loads((ROOT / 'tests/fixtures/trajectory-rebranches-reference.json').read_text())['cases']
    latest = json.loads((ROOT / 'tests/fixtures/parent-position-rebranches-reference.json').read_text())['cases']
    parent_root = json.loads((ROOT / 'tests/fixtures/parent-probabilities-reference.json').read_text())['root']
    indices = [0, 1, 2, 3, 6, 7, 12, 13, 18, 19, 20, 21, 22, 23]
    cache = {}

    def predict(ids, edits):
        key = json.dumps([ids, edits], sort_keys=True)
        if key not in cache:
            logits, _ = model(torch.tensor([ids]), edits, capture_activations=False)
            cache[key] = logits[0, -1].softmax(-1)
        return cache[key]

    cases = []
    with torch.no_grad():
        for reference_index in indices:
            example = latest[reference_index]
            workspace = copy.deepcopy(initial)
            prior = parent_root if example['referenceIndex'] == -1 else previous[example['referenceIndex']]
            for side in range(2):
                workspace['nodes'].append(dict(id=f'parent-child-{side}', parentId=None if example['referenceIndex'] == -1 else prior['nodeIds'][side], plan=prior['plans'][side], frame=prior['frames'][side]))
                workspace['nodes'].append(dict(id=f'lineage-latest-{side}', parentId=f'parent-child-{side}', plan=example['plans'][side], frame=example['frames'][side]))
            nodes = [next(node for node in workspace['nodes'] if node['id'] == f'lineage-latest-{side}') for side in range(2)]
            origins = []
            for child in nodes:
                node, links = child, []
                while True:
                    source = node['plan']['source']
                    links.insert(0, dict(nodeId=node['id'], parentId=node['parentId'], sourceBranch=source['branch'], sourceStep=source['step']))
                    if node['parentId'] is None:
                        origins.append(dict(branch=source['branch'], links=links))
                        break
                    node = next(ancestor for ancestor in workspace['nodes'] if ancestor['id'] == node['parentId'])
            root = workspace['root']
            for branch in ['original', 'edited']:
                horizon = max(len(node['frame'][branch]['choices']) for node in nodes)
                for target in [471, 1]:
                    tracks = []
                    for side, node in enumerate(nodes):
                        source = node['plan']['source']
                        parent = next(ancestor for ancestor in workspace['nodes'] if ancestor['id'] == node['parentId'])
                        rows = {}
                        for role in ['original', 'parent', 'child']:
                            path = origins[side]['branch'] if role == 'original' else source['branch'] if role == 'parent' else branch
                            frame = root['frame'][path] if role == 'original' else (parent if role == 'parent' else node)['frame'][path]
                            start = source['step'] - 1 if role == 'original' else source['step'] - parent['plan']['source']['step'] if role == 'parent' else 0
                            points = []
                            for offset in range(min(horizon, max(0, len(frame['choices']) - start))):
                                index = start + offset
                                prefix_length = len(root['tokens']) + index if role == 'original' else len((parent if role == 'parent' else node)['plan']['source']['prefix']) + index
                                prefix = frame['tokenIds'][:prefix_length]
                                edits = ([root['edit']] if path == 'edited' else []) if role == 'original' else edits_at((parent if role == 'parent' else node)['plan'], path == 'edited', index)
                                choice = frame['choices'][index]
                                points.append(dict(offset=offset, step=source['step'] + offset, targetTokenId=target, probability=float(predict(prefix, edits)[target]), prefix=prefix, edits=edits, tokenId=choice['tokenId'], recordedProbability=choice['probability']))
                            rows[role] = points
                        tracks.append(rows)
                    cases.append(dict(rebranchReferenceIndex=reference_index, branch=branch, targetTokenId=target, origins=origins, rows=tracks))
    (ROOT / 'tests/fixtures/lineage-probabilities-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2) + '\n')
    print(f'Exported {len(cases)} independent three-version probability traces; {len(cache)} distributions.')


if __name__ == '__main__':
    main()
