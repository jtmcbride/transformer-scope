"""Generate reviewed reference artifacts; never executed implicitly by tests."""
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F

from reference_model import ReferenceTransformer, configuration, serialized_tensor

ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
torch.set_num_threads(1)
torch.manual_seed(SEED)
model = ReferenceTransformer()
parameters = model.exported_parameters()
weights = {name: serialized_tensor(p, transpose) for name, (p, transpose) in parameters.items()}

def patch(layer, head, factor, positions=None):
    return {"id": f"{layer}-{head}-{factor}-{positions}", "kind": "head.scale", "layer": layer, "head": head,
            "factor": factor, "tokens": {"kind": "all"} if positions is None else {"kind": "indices", "indices": positions}}

cases = []
for length in (1, 2, 5, 8):
    tokens = [1, 5, 9, 3, 7, 2, 11, 4][:length]
    selections = [[], *[[patch(l, h, factor)] for l in (0, 1) for h in (0, 1) for factor in (0, .5, 1, 2)],
                  [patch(0, 0, .5, [0])], [patch(1, 1, 0, [length-1])],
                  [patch(0, 0, .5, [0]), patch(0, 1, 2, [length-1])],
                  [patch(0, 0, .5), patch(1, 1, 0)]]
    for number, patches in enumerate(selections):
        with torch.no_grad():
            logits, activations = model(torch.tensor([tokens]), patches)
        cases.append({"name": f"length-{length}-case-{number}", "tokenIds": tokens, "interventions": patches,
                      "activations": {name: serialized_tensor(value) for name, value in activations.items()}})

train_tokens = torch.tensor([[1, 5, 9, 3, 7]])
targets = torch.tensor([[5, 9, 3, 7, 2]])
# The key bias has an analytically zero gradient (softmax is shift invariant).
# A 1e-8 epsilon amplifies independent float32 roundoff into nontrivial updates.
# Use an explicit, conditioned optimizer probe without loosening parity bounds.
optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-5)
logits, _ = model(train_tokens)
loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
loss.backward()
gradients = {name: serialized_tensor(p.grad, transpose) for name, (p, transpose) in parameters.items()}
optimizer.step()
updated = {name: serialized_tensor(p, transpose) for name, (p, transpose) in parameters.items()}

binary = bytearray()
entries = []
for name, tensor in weights.items():
    payload = np.asarray(tensor["values"], dtype="<f4").tobytes()
    entries.append({"name": name, "shape": tensor["shape"], "byteOffset": len(binary), "byteLength": len(payload)})
    binary.extend(payload)
tokenizer = {"schemaVersion": 1, "kind": "symbols", "bosTokenId": 0,
             "tokens": ["<BOS>", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]}
tokenizer_bytes = (json.dumps(tokenizer, indent=2) + "\n").encode()
manifest = {"schemaVersion": 1, "architectureVersion": "preln-absolute-tied-v1", "modelId": "verification-seed-1729",
            "config": configuration(model), "dtype": "float32-le", "matrixOrientation": "input-output",
            "conventions": {"gelu": "tanh", "layerNormVariance": "population", "position": "learned-absolute",
                            "tiedEmbeddings": True, "dropout": 0, "maskValue": -1e9},
            "weightsFile": "weights.bin", "tokenizerFile": "tokenizer.json", "tensors": entries,
            "weightsSha256": hashlib.sha256(binary).hexdigest(), "tokenizerSha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "provenance": {"purpose": "untrained numerical verification", "seed": SEED, "torchVersion": torch.__version__, "generatorVersion": 1}}
manifest["modelRevision"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
fixture = {"schemaVersion": 1, "metadata": manifest["provenance"], "manifest": manifest, "weights": weights,
           "cases": cases, "training": {"tokenIds": train_tokens[0].tolist(), "targets": targets[0].tolist(), "loss": loss.item(),
           "optimizer": {"kind": "adam", "learningRate": .001, "beta1": .9, "beta2": .999, "epsilon": 1e-5},
           "gradients": gradients, "updatedWeights": updated}}
fixture_path = ROOT / "tests/fixtures/reference.json"
fixture_path.parent.mkdir(parents=True, exist_ok=True)
fixture_path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
output = ROOT / "public/models/verification"
output.mkdir(parents=True, exist_ok=True)
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(output / "tokenizer.json").write_bytes(tokenizer_bytes)
(output / "weights.bin").write_bytes(binary)
(output / "probe.json").write_text(json.dumps({"cases": [case for case in cases if case["name"] in
    ("length-5-case-0", "length-5-case-2")], "training": fixture["training"]}, separators=(",", ":")) + "\n")
print(f"Exported {len(cases)} cases, {len(weights)} tensors, {len(binary)} weight bytes to {output}")
