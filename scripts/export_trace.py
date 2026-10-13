"""Independent PyTorch stage measurements and diagnostic readouts for effect tracing."""
import json
import numpy as np
import torch
from artifact_model import load_artifact_model, root

torch.set_num_threads(1)
STAGES = ('block.input', 'attention.normalized', 'attention.weights', 'attention.head.raw', 'attention.head.modified',
          'attention.projected', 'attention.residual', 'mlp.output', 'block.output')

def measurements(model, capture, other, length):
    refs = [('embedding.output', None), ('final.normalized', None), ('logits', None)] + [(stage, layer) for layer in range(model.config.layers) for stage in STAGES]
    rows = []
    for stage, layer in refs:
        key = stage if layer is None else f'{layer}:{stage}'
        a, b = capture[key][0].double().numpy(), other[key][0].double().numpy()
        headed = stage in ('attention.weights', 'attention.head.raw', 'attention.head.modified')
        for position in range(length):
            for head in range(model.config.heads if headed else 1):
                x, y = (a[head, position], b[head, position]) if stage == 'attention.weights' else (a[position, head], b[position, head]) if headed else (a[position], b[position])
                rows.append(dict(stage=stage, **({} if layer is None else dict(layer=layer)), position=position, **(dict(head=head) if headed else {}),
                                 baselineNorm=float(np.linalg.norm(x)), modifiedNorm=float(np.linalg.norm(y)),
                                 maxDifference=float(np.max(np.abs(x-y))), differenceNorm=float(np.linalg.norm(x-y))))
    return rows

def case(model, manifest, tokens, patch, position, name):
    with torch.no_grad():
        base, baseline = model(torch.tensor([tokens]))
        modified, changed = model(torch.tensor([tokens]), [patch])
        refs = [('embedding.output', None)] + [(stage, layer) for layer in range(model.config.layers) for stage in ('attention.residual', 'block.output')]
        def read(capture):
            rows = torch.stack([capture[stage if layer is None else f'{layer}:{stage}'][0, position] for stage, layer in refs])
            return (model.finalNorm(rows) @ model.token.weight.T).tolist()
        result = dict(name=name, modelRevision=manifest['modelRevision'], tokens=tokens, intervention=patch, predictionPosition=position,
                    nodes=measurements(model, baseline, changed, len(tokens)), baselineReadouts=read(baseline), modifiedReadouts=read(changed),
                    baselineLogits=base[0, position].tolist(), modifiedLogits=modified[0, position].tolist())
        if patch['kind'] == 'attention.override':
            def row(capture, stage):
                return capture[f"{patch['layer']}:{stage}"][0, patch['head'], patch['query']].tolist()
            result['attentionRow'] = dict(baselineWeights=row(baseline, 'attention.weights'), modifiedWeights=row(changed, 'attention.weights'),
                                          baselineScores=row(baseline, 'attention.scores'), modifiedScores=row(changed, 'attention.scores'))
        return result

if __name__ == '__main__':
    retrieval, retrieval_manifest = load_artifact_model('retrieval')
    recipes = json.loads((root / 'public/models/retrieval/recipes.json').read_text())['recipes']
    cases = [case(retrieval, retrieval_manifest, r['tokenIds'], r['intervention'], r['predictionPosition'], r['id']) for r in recipes]
    verification, verification_manifest = load_artifact_model('verification')
    for name, layer, position, factor, prediction in [('identity', 0, 5, 1., 5), ('future', 0, 5, 0., 2), ('same-final', 1, 5, .37, 5), ('early-reachable', 0, 0, 0., 5)]:
        patch = dict(id=name, kind='head.scale', layer=layer, head=0, factor=factor, tokens=dict(kind='indices', indices=[position]))
        cases.append(case(verification, verification_manifest, [0, 1, 5, 9, 3, 7], patch, prediction, name))
    (root / 'tests/fixtures/trace-reference.json').write_text(json.dumps(dict(torchVersion=torch.__version__, cases=cases), indent=2) + '\n')
    print(f'Exported {len(cases)} trace cases from hash-checked shipped weights.')
