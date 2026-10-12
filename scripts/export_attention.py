"""Independent attention.override fixtures: verification-model cases and a retrieval routing sweep."""
import json

import torch
from artifact_model import load_artifact_model, root

torch.set_num_threads(1)


def override(layer, head, query, weights, name=None):
    return dict(id=name or f'attention-{layer}-{head}-{query}', kind='attention.override', layer=layer, head=head, query=query, weights=[float(w) for w in weights])


def run(model, tokens, patches=()):
    with torch.no_grad():
        return model(torch.tensor([tokens]), list(patches))


def one_hot(length, position):
    return [1. if i == position else 0. for i in range(length)]


verification, manifest = load_artifact_model('verification')
tokens = [0, 1, 5, 9, 3, 7]
uniform = [1.] * 4 + [0.] * 2
specs = [
    ('one-hot-first-layer', [override(0, 1, 3, one_hot(6, 0))]),
    ('uniform-unnormalized', [override(0, 0, 3, uniform)]),
    ('scaled-weights', [override(1, 1, 5, [3., 0., 1., 0., 2., 4.])]),
    ('final-layer-earlier-query', [override(1, 0, 2, one_hot(6, 1))]),
    ('with-head-scale-and-replacement', [dict(id='head', kind='head.scale', layer=0, head=1, factor=.25, tokens=dict(kind='all')),
                                         dict(id='replace', kind='residual.replace', layer=0, stage='block.input', position=1, values=[.5] * 8),
                                         override(0, 1, 4, one_hot(6, 1)), override(0, 0, 4, [1., 1., 0., 0., 0., 0.])]),
]
cases = []
baseline_logits, _ = run(verification, tokens)
for name, patches in specs:
    logits, capture = run(verification, tokens, patches)
    cases.append(dict(name=name, tokenIds=tokens, interventions=patches, logits=logits[0].tolist(),
                      stages={f'{layer}:{stage}': capture[f'{layer}:{stage}'].flatten().tolist() for layer in range(2)
                              for stage in ('attention.scores', 'attention.weights', 'attention.head.raw', 'attention.residual', 'block.output')}))
assert torch.equal(torch.tensor(cases[3]['logits'])[-1], baseline_logits[0, -1]), 'A final-layer override at an earlier query cannot reach the last position.'

retrieval, retrieval_manifest = load_artifact_model('retrieval')
symbols = json.loads((root / 'public/models/retrieval/tokenizer.json').read_text())['tokens']
text = 'A = 0 ; B = 3 ; C = 0 ; ? B ='
target = [0] + [symbols.index(symbol) for symbol in text.split()]
base, base_capture = run(retrieval, target)
sweep = []
for head in range(retrieval.config.heads):
    for position in range(len(target)):
        logits, capture = run(retrieval, target, [override(1, head, 15, one_hot(len(target), position))])
        sweep.append(dict(layer=1, head=head, query=15, position=position, logits=logits[0, -1].tolist()))
result = dict(verificationRevision=manifest['modelRevision'], retrievalRevision=retrieval_manifest['modelRevision'], torchVersion=torch.__version__, cases=cases,
              routing=dict(text=text, tokenIds=target, predictionPosition=15, baselineLogits=base[0, -1].tolist(),
                           baselineRows={str(h): base_capture['1:attention.weights'][0, h, 15].tolist() for h in range(retrieval.config.heads)}, sweep=sweep))
(root / 'tests/fixtures/attention-reference.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Exported {len(cases)} verification cases and {len(sweep)} retrieval routing cases.')
