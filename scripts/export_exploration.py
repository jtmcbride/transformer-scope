"""Independent residual-stream read-out ("logit lens") and embedding values for the trained retrieval checkpoint.

Each residual is passed through the final LayerNorm and the tied unembedding. The model computes
this only for the final residual; intermediate read-outs are an inspection, not model predictions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from reference_model import Config, ReferenceTransformer

root = Path(__file__).resolve().parents[1]
directory = root / 'public/models/retrieval'
manifest = json.loads((directory / 'manifest.json').read_text())
payload = (directory / 'weights.bin').read_bytes()
assert hashlib.sha256(payload).hexdigest() == manifest['weightsSha256'], 'Weights do not match the manifest.'
config = {key: manifest['config'][key] for key in Config.__dataclass_fields__}
model = ReferenceTransformer(Config(**config))
parameters = model.exported_parameters()
with torch.no_grad():
    for entry in manifest['tensors']:
        values = np.frombuffer(payload, dtype='<f4', count=entry['byteLength'] // 4, offset=entry['byteOffset']).reshape(entry['shape'])
        parameter, transpose = parameters[entry['name']]
        parameter.copy_(torch.from_numpy(values.T.copy() if transpose else values.copy()))
assert set(parameters) == {entry['name'] for entry in manifest['tensors']}
recipes = json.loads((directory / 'recipes.json').read_text())['recipes']
recipe = next(item for item in recipes if item['id'] == 'effect')
tokens, position = recipe['tokenIds'], recipe['predictionPosition']
stages = [('embedding.output', None)] + [(stage, layer) for layer in range(model.config.layers) for stage in ('attention.residual', 'block.output')]

def lens(capture):
    key = lambda stage, layer: stage if layer is None else f'{layer}:{stage}'
    residuals = torch.stack([capture[key(stage, layer)][0, position] for stage, layer in stages])
    return dict(norms=residuals.norm(dim=-1).tolist(), logits=(model.finalNorm(residuals) @ model.token.weight.T).tolist())

torch.set_num_threads(1)
with torch.no_grad():
    logits, baseline = model(torch.tensor([tokens]))
    modified_logits, modified = model(torch.tensor([tokens]), [recipe['intervention']])
# The loaded weights must reproduce the recorded recipe before any read-out is trusted.
assert np.allclose(logits[0, position].numpy(), recipe['metrics']['baselineLogits'], atol=2e-5, rtol=1e-5)
assert np.allclose(modified_logits[0, position].numpy(), recipe['metrics']['modifiedLogits'], atol=2e-5, rtol=1e-5)
result = dict(modelRevision=manifest['modelRevision'], torchVersion=torch.__version__, tokenIds=tokens, predictionPosition=position,
              intervention=recipe['intervention'], stages=[dict(stage=stage, layer=layer) for stage, layer in stages],
              baseline=lens(baseline), modified=lens(modified),
              finalLogits=dict(baseline=logits[0, position].tolist(), modified=modified_logits[0, position].tolist()))
(root / 'tests/fixtures/lens.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Exported read-outs for {len(stages)} residual stages, baseline and modified.')

# Embedding rows at selected positions, and each token row's cosine similarity to every token embedding.
with torch.no_grad():
    table = model.token.weight
    unit = table / table.norm(dim=-1, keepdim=True)
    embedding = []
    for position in (0, 13, 14, 15):
        token_id = tokens[position]
        token, place = table[token_id], model.position.weight[position]
        embedding.append(dict(position=position, tokenId=token_id, token=token.tolist(), positionEmbedding=place.tolist(),
                              sum=baseline['embedding.output'][0, position].tolist(), similarity=(unit @ unit[token_id]).tolist()))
(root / 'tests/fixtures/embedding.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], torchVersion=torch.__version__,
                                                                   tokenIds=tokens, positions=embedding), indent=2) + '\n')
print(f'Exported embedding rows and similarities at {len(embedding)} positions.')
