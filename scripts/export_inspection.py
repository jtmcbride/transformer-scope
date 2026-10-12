"""Independent values for the browser's BOS-prefixed input and stage selectors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from reference_model import ReferenceTransformer, serialized_tensor

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.manual_seed(1729)
model = ReferenceTransformer()
manifest = json.loads((root / 'public/models/verification/manifest.json').read_text())
payload = b''.join(np.asarray(serialized_tensor(p, transpose)['values'], dtype='<f4').tobytes()
                   for p, transpose in model.exported_parameters().values())
assert hashlib.sha256(payload).hexdigest() == manifest['weightsSha256'], 'Seed weights do not match the installed checkpoint.'
tokens = [0, 1, 5, 9, 3, 7]
patch = dict(id='head-strength', kind='head.scale', layer=0, head=0, factor=.37, tokens=dict(kind='all'))
with torch.no_grad():
    _, baseline = model(torch.tensor([tokens]))
    _, modified = model(torch.tensor([tokens]), [patch])
stages = ['attention.projected', 'attention.residual', 'mlp.output', 'block.output']
cells = []
for query, source in [(2, 1), (0, 3)]:
    cells.append(dict(query=query, source=source, q=baseline['0:attention.query'][0, query, 0].tolist(),
                      k=baseline['0:attention.key'][0, source, 0].tolist(), score=baseline['0:attention.scores'][0, 0, query, source].item(),
                      maskedScore=baseline['0:attention.maskedScores'][0, 0, query, source].item(), weight=baseline['0:attention.weights'][0, 0, query, source].item()))
result = dict(modelRevision=manifest['modelRevision'], torchVersion=torch.__version__, tokenIds=tokens, intervention=patch, cells=cells,
              vectors={str(layer): {stage: dict(baseline=baseline[f'{layer}:{stage}'][0, -1].tolist(), modified=modified[f'{layer}:{stage}'][0, -1].tolist()) for stage in stages} for layer in range(2)})
(root / 'tests/fixtures/inspection.json').write_text(json.dumps(result, indent=2) + '\n')
print('Exported independent attention cells and eight baseline/modified stage pairs.')
