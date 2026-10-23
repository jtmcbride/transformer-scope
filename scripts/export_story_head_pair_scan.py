"""Independent complete pair scans, including mixed-strength assignments and inherited edits."""
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
    edited = json.loads((ROOT/'tests/fixtures/story-forks-reference.json').read_text())['cases'][2]['plan']['source']
    sources = json.loads((ROOT/'tests/fixtures/story-head-scan-reference.json').read_text())['cases']
    cases = []
    heads = list(itertools.product(range(model.config.layers), range(model.config.heads)))
    def edit(head, factor):
        return dict(id='head-scan', kind='head.scale', layer=head[0], head=head[1], factor=factor, tokens=dict(kind='all'))
    with torch.no_grad():
        for source, strengths, factors in [(sources[0]['source'], 'mute', [(0, 0)]),
                                           (sources[1]['source'], 'mute', [(0, 0)]),
                                           (sources[0]['source'], 'opposite', [(0, 2), (2, 0)]),
                                           (edited, 'opposite', [(0, 2), (2, 0)])]:
            rows = []
            for a, b in itertools.combinations(heads, 2):
                for first, second in factors:
                    edits = [edit(a, first), edit(b, second)]
                    plan = dict(source=source, edit=edits[0], additionalEdits=[edits[1]], duration='prediction', maxTokens=1)
                    logits, _ = model(torch.tensor([source['prefix']]), edits_at(plan, True, 0), capture_activations=False)
                    values = logits[0, -1]
                    order = sorted([i for i in range(len(values)) if i not in (0, 2)], key=lambda i: (-float(values[i]), i))
                    probability = float(values.softmax(-1)[source['choice']['tokenId']])
                    rows.append(dict(edits=edits, after=probability, rank=order.index(source['choice']['tokenId'])+1, choice=order[0]))
            best_row = max(rows, key=lambda row: abs(row['after']-source['choice']['probability']))
            best_plan = dict(source=source, edit=best_row['edits'][0], additionalEdits=[best_row['edits'][1]], duration='continuation', maxTokens=6)
            best = dict(plan=best_plan)
            for branch in ['original', 'edited']:
                ids, choices = list(source['prefix']), []
                for i in range(6):
                    logits, _ = model(torch.tensor([ids]), edits_at(best_plan, branch == 'edited', i), capture_activations=False)
                    values = logits[0,-1]
                    masked = values.clone(); masked[0] = masked[2] = float('-inf')
                    token = int(masked.argmax())
                    choices.append(dict(tokenId=token, probability=float(values.softmax(-1)[token])))
                    ids.append(token)
                    if token == 1 or len(ids) >= model.config.contextLength:
                        break
                best[branch] = dict(tokenIds=ids, choices=choices)
            cases.append(dict(source=source, strengths=strengths, rows=rows, best=best))
    (ROOT/'tests/fixtures/story-head-pair-scan-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2)+'\n')
    print(f'Exported {sum(len(case["rows"]) for case in cases)} joint predictions across {len(cases)} full pair scans.')


if __name__ == '__main__':
    main()
