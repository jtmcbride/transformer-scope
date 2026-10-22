"""Independent joint and individual predictions, alternatives, and rollouts for two-head forks."""
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
    root = json.loads((ROOT/'tests/fixtures/story-forks-reference.json').read_text())['cases']
    nested = json.loads((ROOT/'tests/fixtures/nested-story-forks-reference.json').read_text())['cases']
    cases = []
    specs = [(root[0]['plan']['source'], (0, 0, 0), (1, 1, 0), 'continuation'),
             (root[0]['plan']['source'], (0, 0, 0), (1, 1, 0), 'prediction'),
             (nested[2]['plan']['source'], (0, 0, 1), (2, 2, 0), 'continuation'),
             (nested[1]['plan']['source'], (1, 1, 0), (2, 2, 0), 'continuation')]
    def edit(spec):
        layer, head, factor = spec
        return dict(id='head-scan', kind='head.scale', layer=layer, head=head, factor=factor, tokens=dict(kind='all'))
    def distribution(ids, edits):
        logits, _ = model(torch.tensor([ids]), edits, capture_activations=False)
        values = logits[0, -1]
        order = sorted([i for i in range(len(values)) if i not in (0, 2)], key=lambda i: (-float(values[i]), i))
        return values.softmax(-1).tolist(), order
    with torch.no_grad():
        for source, a, b, duration in specs:
            plan = dict(source=copy.deepcopy(source), edit=edit(a), additionalEdits=[edit(b)], duration=duration, maxTokens=6)
            before, ar = distribution(source['prefix'], edits_at(plan, False, 0))
            after, br = distribution(source['prefix'], edits_at(plan, True, 0))
            selected = source['choice']['tokenId']
            individuals = []
            for proposed in [plan['edit'], *plan['additionalEdits']]:
                alone = dict(plan, edit=proposed, additionalEdits=[])
                probs, _ = distribution(source['prefix'], edits_at(alone, True, 0))
                individuals.append(dict(edit=proposed, probability=probs[selected], delta=probs[selected]-before[selected]))
            ids = {selected, *ar[:5], *br[:5], ar[0], br[0]}
            rows = [dict(tokenId=i, before=before[i], after=after[i], delta=after[i]-before[i], beforeRank=ar.index(i)+1, afterRank=br.index(i)+1) for i in sorted(ids)]
            report = dict(rows=rows, beforeChoice=ar[0], afterChoice=br[0], individuals=individuals, interaction=after[selected]-before[selected]-sum(row['delta'] for row in individuals))
            result = dict(plan=plan, report=report)
            for name in ['original', 'edited']:
                token_ids, choices = list(source['prefix']), []
                for i in range(plan['maxTokens']):
                    probs, order = distribution(token_ids, edits_at(plan, name == 'edited', i))
                    token = order[0]
                    choices.append(dict(tokenId=token, probability=probs[token]))
                    token_ids.append(token)
                    if token == 1 or len(token_ids) >= model.config.contextLength:
                        break
                result[name] = dict(tokenIds=token_ids, choices=choices)
            cases.append(result)
    (ROOT/'tests/fixtures/story-head-pairs-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2)+'\n')
    print(f'Exported {len(cases)} independent two-head comparisons and rollouts.')


if __name__ == '__main__':
    main()
