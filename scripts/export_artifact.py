"""Reusable, deterministic float32 artifact and independent verification-probe exporter."""
import copy
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from reference_model import configuration, serialized_tensor


def write_artifact(model, output, symbols, model_id, provenance, tokenizer_bytes=None):
    """symbols: a symbol list for a format 1 tokenizer, or None when tokenizer_bytes holds a serialized tokenizer."""
    output.mkdir(parents=True, exist_ok=True)
    binary, entries = bytearray(), []
    for name, (parameter, transpose) in model.exported_parameters().items():
        record = serialized_tensor(parameter, transpose)
        payload = np.asarray(record['values'], dtype='<f4').tobytes()
        entries.append(dict(name=name, shape=record['shape'], byteOffset=len(binary), byteLength=len(payload)))
        binary.extend(payload)
    if tokenizer_bytes is None:
        tokenizer = dict(schemaVersion=1, kind='symbols', bosTokenId=0, tokens=symbols)
        tokenizer_bytes = (json.dumps(tokenizer, indent=2) + '\n').encode()
    manifest = dict(schemaVersion=1, architectureVersion='preln-absolute-tied-v1', modelId=model_id,
                    config=configuration(model), dtype='float32-le', matrixOrientation='input-output',
                    conventions=dict(gelu='tanh', layerNormVariance='population', position='learned-absolute',
                                     tiedEmbeddings=True, dropout=0, maskValue=-1e9),
                    weightsFile='weights.bin', tokenizerFile='tokenizer.json', tensors=entries,
                    weightsSha256=hashlib.sha256(binary).hexdigest(), tokenizerSha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
                    provenance=provenance)
    manifest['modelRevision'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'tokenizer.json').write_bytes(tokenizer_bytes)
    (output / 'weights.bin').write_bytes(binary)
    return manifest


def write_probe(model, output, tokens, interventions):
    # A separate clone ensures probe Adam does not change the exported checkpoint.
    probe_model = copy.deepcopy(model)
    cases = []
    for index, patches in enumerate([[], interventions]):
        with torch.no_grad():
            _, capture = probe_model(torch.tensor([tokens]), patches)
        cases.append(dict(name=f'trained-case-{index}', tokenIds=list(tokens), interventions=patches,
                          activations={name: serialized_tensor(value) for name, value in capture.items()}))
    # This numerical probe uses the existing full-sequence loss contract, independently of
    # the checkpoint's final-answer-only training objective.
    targets = list(tokens[1:]) + [5]
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=.001, betas=(.9, .999), eps=1e-5)
    logits, _ = probe_model(torch.tensor([tokens]), capture_activations=False)
    loss = F.cross_entropy(logits.reshape(-1, model.config.vocabularySize), torch.tensor(targets))
    loss.backward()
    gradients = {name: serialized_tensor(p.grad, transpose) for name, (p, transpose) in probe_model.exported_parameters().items()}
    optimizer.step()
    updated = {name: serialized_tensor(p, transpose) for name, (p, transpose) in probe_model.exported_parameters().items()}
    training = dict(tokenIds=list(tokens), targets=targets, loss=loss.item(),
                    optimizer=dict(kind='adam', learningRate=.001, beta1=.9, beta2=.999, epsilon=1e-5),
                    gradients=gradients, updatedWeights=updated)
    (output / 'probe.json').write_text(json.dumps(dict(cases=cases, training=training), separators=(',', ':')) + '\n')
