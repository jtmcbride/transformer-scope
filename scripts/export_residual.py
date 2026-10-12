"""Independent residual.replace fixtures: verification-model cases and a retrieval patching sweep."""
import json

import torch
from artifact_model import load_artifact_model, root

torch.set_num_threads(1)
HOOKS = ('block.input', 'attention.residual')


def replace(layer, stage, position, values, name=None):
    return dict(id=name or f'{stage}-{layer}-{position}', kind='residual.replace', layer=layer, stage=stage, position=position, values=[float(v) for v in values])


def run(model, tokens, patches=()):
    with torch.no_grad():
        logits, capture = model(torch.tensor([tokens]), list(patches))
    return logits, capture


def stages(model, capture):
    return {f'{layer}:{stage}': capture[f'{layer}:{stage}'][0].tolist() for layer in range(model.config.layers) for stage in (*HOOKS, 'block.output')}


verification, manifest = load_artifact_model('verification')
generator = torch.Generator().manual_seed(2027)
vector = lambda scale=1.: (torch.randn(verification.config.modelDim, generator=generator) * scale).tolist()
tokens, short = [0, 1, 5, 9, 3, 7], [0, 4, 2, 8]
_, plain = run(verification, tokens)
head = dict(id='head', kind='head.scale', layer=0, head=1, factor=.5, tokens=dict(kind='all'))
specs = [
    ('block-input-0', tokens, [replace(0, 'block.input', 2, vector())]),
    ('block-input-1-first', tokens, [replace(1, 'block.input', 0, vector())]),
    ('attention-residual-0-last', tokens, [replace(0, 'attention.residual', 5, vector())]),
    ('attention-residual-1-unreachable', tokens, [replace(1, 'attention.residual', 1, vector())]),
    ('combined-with-head-scale', tokens, [head, replace(0, 'attention.residual', 3, vector()), replace(1, 'block.input', 4, vector(2.))]),
    ('zero-vector', short, [replace(0, 'block.input', 0, [0.] * verification.config.modelDim)]),
    ('baseline-values', tokens, [replace(0, 'attention.residual', 2, plain['0:attention.residual'][0, 2].tolist())]),
]
cases = []
for name, ids, patches in specs:
    logits, capture = run(verification, ids, patches)
    cases.append(dict(name=name, tokenIds=ids, interventions=patches, logits=logits[0].tolist(), stages=stages(verification, capture)))
baseline_logits, _ = run(verification, tokens)
assert torch.equal(torch.tensor(cases[-1]['logits']), baseline_logits[0]), 'Replacing a row with its own values must not change logits.'
assert torch.equal(torch.tensor(cases[3]['logits'])[-1], baseline_logits[0, -1]), 'A final-layer earlier-position replacement cannot reach the last position.'

# Retrieval patching sweep: the inputs differ only in the queried key (position 14).
retrieval, retrieval_manifest = load_artifact_model('retrieval')
symbols = json.loads((root / 'public/models/retrieval/tokenizer.json').read_text())['tokens']
encode = lambda text: [0] + [symbols.index(symbol) for symbol in text.split()]
target_text, source_text = 'A = 0 ; B = 3 ; C = 0 ; ? B =', 'A = 0 ; B = 3 ; C = 0 ; ? A ='
target, source = encode(target_text), encode(source_text)
assert [i for i, (a, b) in enumerate(zip(target, source)) if a != b] == [14]
_, source_capture = run(retrieval, source)
target_logits, _ = run(retrieval, target)
sweep = []
for layer in range(retrieval.config.layers):
    for stage in HOOKS:
        for position in range(len(target)):
            patch = replace(layer, stage, position, source_capture[f'{layer}:{stage}'][0, position].tolist())
            logits, _ = run(retrieval, target, [patch])
            sweep.append(dict(layer=layer, stage=stage, position=position, logits=logits[0, -1].tolist()))
result = dict(verificationRevision=manifest['modelRevision'], retrievalRevision=retrieval_manifest['modelRevision'], torchVersion=torch.__version__,
              cases=cases, patching=dict(targetText=target_text, sourceText=source_text, targetTokenIds=target, sourceTokenIds=source,
                                         predictionPosition=len(target) - 1, baselineLogits=target_logits[0, -1].tolist(), sweep=sweep))
(root / 'tests/fixtures/residual-reference.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Exported {len(cases)} verification cases and {len(sweep)} retrieval patching cases.')
