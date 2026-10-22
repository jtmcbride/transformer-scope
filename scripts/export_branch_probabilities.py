"""Independent same-prefix head-change probabilities from the shipped PyTorch weights."""
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
model, _, manifest = load_model(ROOT / 'public/models/tinystories')
cases = []
for file in ['story-forks-reference.json', 'nested-story-forks-reference.json']:
    for item in json.loads((ROOT / 'tests/fixtures' / file).read_text())['cases']:
        plan = item['plan']
        distributions, ranks, choices = [], [], []
        with torch.no_grad():
            for changed in [False, True]:
                logits, _ = model(torch.tensor([plan['source']['prefix']]), edits_at(plan, changed, 0), capture_activations=False)
                values = logits[0, -1]
                order = sorted([i for i in range(len(values)) if i not in (0, 2)], key=lambda i: (-float(values[i]), i))
                distributions.append(values.softmax(-1).tolist())
                ranks.append(order)
                choices.append(order[0])
        selected = plan['source']['choice']['tokenId']
        ids = {selected, *ranks[0][:5], *ranks[1][:5], *choices}
        rows = [dict(tokenId=i, before=distributions[0][i], after=distributions[1][i], delta=distributions[1][i]-distributions[0][i], beforeRank=ranks[0].index(i)+1, afterRank=ranks[1].index(i)+1) for i in ids]
        cases.append(dict(plan=plan, beforeChoice=choices[0], afterChoice=choices[1], rows=rows))
(ROOT / 'tests/fixtures/branch-probabilities-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2)+'\n')
print(f'Exported {len(cases)} branch-point probability comparisons.')
