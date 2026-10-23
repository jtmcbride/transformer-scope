"""Independent four-path continuations for one head pair applied to two captured sources."""
import copy
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at
ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model, _, manifest = load_model(ROOT/'public/models/tinystories')
    maps = json.loads((ROOT/'tests/fixtures/story-heatmap-comparison-reference.json').read_text())
    nested = json.loads((ROOT/'tests/fixtures/nested-story-forks-reference.json').read_text())['cases']
    strongest = max(maps['differences'], key=lambda row: abs(row['difference']))['edits']
    sources = [case['source'] for case in maps['cases']]
    mixed = [dict(id='head-scan', kind='head.scale', layer=0, head=0, factor=1, tokens=dict(kind='all')),
             dict(id='head-scan', kind='head.scale', layer=2, head=2, factor=2, tokens=dict(kind='all'))]
    identity = copy.deepcopy(mixed)
    identity[1]['factor'] = 1
    specs = [(sources, strongest, 'continuation', 6), (sources, strongest, 'prediction', 6),
             ([nested[1]['plan']['source'], nested[2]['plan']['source']], mixed, 'continuation', 4),
             ([sources[0], sources[0]], identity, 'continuation', 6)]
    cases = []
    with torch.no_grad():
        for selected, pair, duration, count in specs:
            plans = [dict(source=copy.deepcopy(source), edit=copy.deepcopy(pair[0]), additionalEdits=[copy.deepcopy(pair[1])], duration=duration, maxTokens=count) for source in selected]
            result = dict(plans=plans, frames=[])
            for plan in plans:
                frame = {}
                for branch in ['original', 'edited']:
                    ids, choices = list(plan['source']['prefix']), []
                    for index in range(count):
                        logits, _ = model(torch.tensor([ids]), edits_at(plan, branch == 'edited', index), capture_activations=False)
                        values = logits[0,-1]
                        eligible = values.clone()
                        eligible[0] = eligible[2] = float('-inf')
                        token = int(eligible.argmax())
                        choices.append(dict(tokenId=token, probability=float(values.softmax(-1)[token])))
                        ids.append(token)
                        if token == 1 or len(ids) >= model.config.contextLength:
                            break
                    reason = 'end of story' if ids[-1] == 1 else 'context limit' if len(ids) >= model.config.contextLength else 'token limit'
                    frame[branch] = dict(tokenIds=ids, choices=choices, finish=reason)
                result['frames'].append(dict(step=max(len(frame['original']['choices']),len(frame['edited']['choices'])), **frame))
            cases.append(result)
    (ROOT/'tests/fixtures/story-shared-forks-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2)+'\n')
    print(f'Exported {len(cases)} independent shared-pair comparisons, each with four continuations.')


if __name__ == '__main__':
    main()
