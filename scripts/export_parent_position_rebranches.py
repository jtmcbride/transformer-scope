"""Independent PyTorch rankings of actual parent/new probabilities and another generation of branches."""
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
    traces = json.loads((ROOT / 'tests/fixtures/parent-probabilities-reference.json').read_text())
    branches = json.loads((ROOT / 'tests/fixtures/trajectory-rebranches-reference.json').read_text())['cases']
    cache = {}

    def predict(ids, edits):
        key = json.dumps([ids, edits], sort_keys=True)
        if key not in cache:
            logits, _ = model(torch.tensor([ids]), edits, capture_activations=False)
            cache[key] = logits[0, -1].clone()
        return cache[key]

    scenarios = [
        (6, 'edited', 471, ['absolute', 'increase', 'decrease'], ['same', 'amplify']),
        (7, 'edited', 1, ['absolute', 'increase', 'decrease'], ['amplify']),
        (14, 'original', 471, ['absolute'], ['amplify']),
        (-1, 'edited', 471, ['absolute'], ['same', 'amplify']),
    ]
    cases = []
    with torch.no_grad():
        for reference_index, branch, target, orders, strengths in scenarios:
            prior = traces['root'] if reference_index == -1 else branches[reference_index]
            trace = next(row for row in traces['cases'] if row['referenceIndex'] == reference_index
                         and row['childBranch'] == branch and row['targetTokenId'] == target)
            count = min(len(side[role]) for side in trace['rows'] for role in ['parent', 'child'])
            positions = []
            for offset in range(count):
                a, b = [side['child'][offset]['probability'] - side['parent'][offset]['probability'] for side in trace['rows']]
                positions.append(dict(offset=offset, a=a, b=b))
            for order in orders:
                def score(row):
                    if order == 'absolute':
                        return max(abs(row['a']), abs(row['b']))
                    return max(row['a'], row['b']) if order == 'increase' else min(row['a'], row['b'])
                ranked = sorted(positions, key=lambda row: (score(row) if order == 'decrease' else -score(row), row['offset']))
                offset = ranked[0]['offset']
                sources = []
                for plan, frame in zip(prior['plans'], prior['frames']):
                    path = frame[branch]
                    sources.append(dict(branch=branch, step=plan['source']['step'] + offset,
                                        prefix=path['tokenIds'][:len(plan['source']['prefix']) + offset],
                                        choice=copy.deepcopy(path['choices'][offset]), settings=copy.deepcopy(plan['source']['settings']),
                                        edits=edits_at(plan, branch == 'edited', offset),
                                        continuationEdits=edits_at(plan, branch == 'edited', offset + 1)))
                for strength in strengths:
                    pair = copy.deepcopy([prior['plans'][0]['edit'], *prior['plans'][0]['additionalEdits']])
                    if strength == 'amplify':
                        for edit in pair:
                            edit['factor'] = 2
                    for duration in ['continuation', 'prediction']:
                        plans = [dict(source=source, edit=pair[0], additionalEdits=[pair[1]], duration=duration, maxTokens=3) for source in sources]
                        frames = []
                        for plan in plans:
                            assert plan['source']['settings']['temperature'] == 0, 'This reference rollout uses the saved greedy sampler.'
                            frame = {}
                            for path in ['original', 'edited']:
                                ids = list(plan['source']['prefix'])
                                choices = []
                                for index in range(3):
                                    values = predict(ids, edits_at(plan, path == 'edited', index))
                                    eligible = values.clone()
                                    eligible[0] = eligible[2] = float('-inf')
                                    token = int(eligible.argmax())
                                    choices.append(dict(tokenId=token, probability=float(values.softmax(-1)[token])))
                                    ids.append(token)
                                    if token == 1 or len(ids) >= model.config.contextLength:
                                        break
                                frame[path] = dict(tokenIds=ids, choices=choices)
                            frames.append(frame)
                        cases.append(dict(referenceIndex=reference_index, branch=branch, targetTokenId=target, order=order,
                                          ranked=ranked, offset=offset, strengths=strength, duration=duration, plans=plans, frames=frames))
    output = dict(modelRevision=manifest['modelRevision'], cases=cases)
    (ROOT / 'tests/fixtures/parent-position-rebranches-reference.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'Exported {len(cases)} independent four-path rebranches from ranked parent/new changes; {len(cache)} distributions.')


if __name__ == '__main__':
    main()
