"""Independent redirect trace fixtures, using hash-checked shipped weights."""
import json
import torch
from artifact_model import load_artifact_model, root
from export_trace import case

cases = []
retrieval, manifest = load_artifact_model('retrieval')
symbols = json.loads((root / 'public/models/retrieval/tokenizer.json').read_text())['tokens']
tokens = [0] + [symbols.index(symbol) for symbol in 'A = 0 ; B = 3 ; C = 0 ; ? B ='.split()]

def add(model, manifest, tokens, name, layer, head, query, weights, prediction):
    patch = dict(id=name, kind='attention.override', layer=layer, head=head, query=query, weights=weights)
    result = case(model, manifest, tokens, patch, prediction, name)
    assert result['attentionRow']['baselineScores'] == result['attentionRow']['modifiedScores'], 'A redirect acts after scores.'
    cases.append(result)

def one_hot(length, source):
    return [1. if position == source else 0. for position in range(length)]

for name, layer, head, query, source in [('redirect-effect', 1, 1, 15, 5), ('redirect-small', 1, 1, 15, 14),
                                        ('redirect-unreachable', 1, 1, 2, 1), ('redirect-early', 0, 3, 3, 0)]:
    add(retrieval, manifest, tokens, name, layer, head, query, one_hot(len(tokens), source), 15)
add(retrieval, manifest, tokens, 'redirect-uniform', 0, 1, 3, [1.] * 4 + [0.] * 12, 15)

verification, manifest = load_artifact_model('verification')
tokens = [0, 1, 5, 9, 3, 7]
add(verification, manifest, tokens, 'redirect-future', 0, 0, 5, one_hot(6, 0), 2)
add(verification, manifest, tokens, 'redirect-normalized', 1, 1, 5, [3., 0., 1., 0., 2., 4.], 5)
add(verification, manifest, tokens, 'redirect-identical-row', 0, 0, 0, one_hot(6, 0), 5)
(root / 'tests/fixtures/attention-trace-reference.json').write_text(json.dumps(dict(torchVersion=torch.__version__, cases=cases), indent=2) + '\n')
print(f'Exported {len(cases)} redirect traces from hash-checked shipped weights.')
