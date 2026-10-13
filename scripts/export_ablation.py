"""Independent single-head and pairwise ablation logits for the ablation explorer (trained retrieval checkpoint)."""
import json

import torch
from artifact_model import load_artifact_model, root

torch.set_num_threads(1)
model, manifest = load_artifact_model('retrieval')
symbols = json.loads((root / 'public/models/retrieval/tokenizer.json').read_text())['tokens']
encode = lambda text: [0] + [symbols.index(symbol) for symbol in text.split()]
ablate = lambda layer, head: dict(id=f'ablate-{layer}-{head}', kind='head.scale', layer=layer, head=head, factor=0., tokens=dict(kind='all'))
heads = [(layer, head) for layer in range(model.config.layers) for head in range(model.config.heads)]
# Pairs: across layers (the two most harmful heads held out) and within one layer.
pairs = [((0, 3), (1, 1)), ((0, 1), (0, 3)), ((1, 0), (1, 1))]
inputs = []
for text in ('A = 0 ; B = 3 ; C = 0 ; ? B =', 'A = 2 ; C = 0 ; B = 3 ; ? A =', 'D = 1 ; A = 1 ; B = 2 ; ? D ='):
    tokens = encode(text)
    with torch.no_grad():
        baseline, _ = model(torch.tensor([tokens]))
        single = [dict(layer=l, head=h, logits=model(torch.tensor([tokens]), [ablate(l, h)])[0][0, -1].tolist()) for l, h in heads]
        joint = [dict(heads=[list(a), list(b)], logits=model(torch.tensor([tokens]), [ablate(*a), ablate(*b)])[0][0, -1].tolist()) for a, b in pairs]
    target = int(baseline[0, -1].argmax())
    inputs.append(dict(text=text, tokenIds=tokens, predictionPosition=len(tokens) - 1, targetTokenId=target, baselineLogits=baseline[0, -1].tolist(), single=single, pairs=joint))
(root / 'tests/fixtures/ablation-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], torchVersion=torch.__version__, inputs=inputs), indent=2) + '\n')
print(f'Exported {len(inputs)} inputs × {len(heads)} single ablations and {len(pairs)} pairs.')
