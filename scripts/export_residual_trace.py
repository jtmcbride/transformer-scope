"""Independent residual-copy trace fixtures, including source input provenance."""
import json
import torch
from artifact_model import load_artifact_model, root
from export_trace import case

cases = []
def add(model, manifest, tokens, source, name, layer, stage, position, prediction):
    with torch.no_grad():
        _, source_capture = model(torch.tensor([source]))
        values = source_capture[f'{layer}:{stage}'][0, position].tolist()
    patch = dict(id=name, kind='residual.replace', layer=layer, stage=stage, position=position, values=values)
    result = case(model, manifest, tokens, patch, prediction, name)
    result['sourceTokens'] = source
    cases.append(result)

retrieval, manifest = load_artifact_model('retrieval')
symbols = json.loads((root / 'public/models/retrieval/tokenizer.json').read_text())['tokens']
encode = lambda text: [0] + [symbols.index(symbol) for symbol in text.split()]
tokens = encode('A = 0 ; B = 3 ; C = 0 ; ? B =')
source = encode('A = 0 ; B = 3 ; C = 0 ; ? A =')
for name, layer, stage, position, origin in [
    ('copy-input-effect', 1, 'block.input', 14, source),
    ('copy-residual-effect', 1, 'attention.residual', 15, source),
    ('copy-local', 1, 'attention.residual', 14, source),
    ('copy-later-handoff', 0, 'attention.residual', 14, source),
    ('copy-identical', 1, 'attention.residual', 15, tokens),
]:
    add(retrieval, manifest, tokens, origin, name, layer, stage, position, 15)

verification, manifest = load_artifact_model('verification')
tokens, source = [0, 1, 5, 9, 3, 7], [0, 2, 5, 9, 3, 8]
for name, layer, stage, position, prediction in [
    ('copy-future', 0, 'block.input', 5, 2),
    ('copy-final-input', 1, 'block.input', 1, 5),
    ('copy-same-final', 1, 'attention.residual', 5, 5),
]:
    add(verification, manifest, tokens, source, name, layer, stage, position, prediction)
(root / 'tests/fixtures/residual-trace-reference.json').write_text(json.dumps(dict(torchVersion=torch.__version__, cases=cases), indent=2) + '\n')
print(f'Exported {len(cases)} residual-copy traces from hash-checked shipped weights.')
