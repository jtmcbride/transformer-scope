"""Load a shipped model artifact into the independent PyTorch reference.

Seeded re-initialization is not bit-reproducible across PyTorch builds, so exporters that check
shipped checkpoints read weights.bin directly and verify its hash.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from reference_model import Config, ReferenceTransformer

root = Path(__file__).resolve().parents[1]


def load_artifact_model(name):
    directory = root / 'public/models' / name
    manifest = json.loads((directory / 'manifest.json').read_text())
    payload = (directory / 'weights.bin').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest['weightsSha256'], f'{name} weights do not match the manifest.'
    model = ReferenceTransformer(Config(**{key: manifest['config'][key] for key in Config.__dataclass_fields__}))
    parameters = model.exported_parameters()
    assert set(parameters) == {entry['name'] for entry in manifest['tensors']}
    with torch.no_grad():
        for entry in manifest['tensors']:
            values = np.frombuffer(payload, dtype='<f4', count=entry['byteLength'] // 4, offset=entry['byteOffset']).reshape(entry['shape'])
            parameter, transpose = parameters[entry['name']]
            parameter.copy_(torch.from_numpy(values.T.copy() if transpose else values.copy()))
    return model, manifest
