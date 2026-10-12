"""Independent PyTorch reference. Native Linear weights are transposed on export."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    layers: int = 2
    heads: int = 2
    modelDim: int = 8
    mlpDim: int = 16
    vocabularySize: int = 16
    contextLength: int = 8
    layerNormEpsilon: float = 1e-5


def replaced(x, layer, stage, patches):
    """residual.replace: overwrite whole residual rows at one hook; other rows are untouched."""
    selected = [p for p in patches if p.get("kind") == "residual.replace" and p["layer"] == layer and p["stage"] == stage]
    if not selected:
        return x
    x = x.clone()
    for patch in selected:
        x[0, patch["position"]] = torch.tensor(patch["values"], dtype=x.dtype)
    return x


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.ln1 = nn.LayerNorm(c.modelDim, eps=c.layerNormEpsilon)
        self.q = nn.Linear(c.modelDim, c.modelDim)
        self.k = nn.Linear(c.modelDim, c.modelDim)
        self.v = nn.Linear(c.modelDim, c.modelDim)
        self.output = nn.Linear(c.modelDim, c.modelDim)
        self.ln2 = nn.LayerNorm(c.modelDim, eps=c.layerNormEpsilon)
        self.mlpIn = nn.Linear(c.modelDim, c.mlpDim)
        self.mlpOut = nn.Linear(c.mlpDim, c.modelDim)

    def forward(self, x, layer, patches, capture):
        c = self.c
        b, t, d = x.shape
        def save(stage, value):
            if capture is not None:
                capture[f"{layer}:{stage}"] = value.detach().clone()
            return value
        x = save("block.input", replaced(x, layer, "block.input", patches))
        u = save("attention.normalized", self.ln1(x))
        def project(linear, stage):
            return save(stage, linear(u).reshape(b, t, c.heads, d // c.heads))
        q = project(self.q, "attention.query")
        k = project(self.k, "attention.key")
        v = project(self.v, "attention.value")
        scores = save("attention.scores", (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) / (d // c.heads) ** .5)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        masked = save("attention.maskedScores", scores.masked_fill(mask, -1e9))
        a = save("attention.weights", masked.softmax(dim=-1))
        z = save("attention.head.raw", (a @ v.transpose(1, 2)).transpose(1, 2).contiguous())
        factors = torch.ones(1, t, c.heads, 1)
        for patch in patches:
            if patch.get("kind", "head.scale") != "head.scale" or patch["layer"] != layer:
                continue
            positions = range(t) if patch["tokens"]["kind"] == "all" else patch["tokens"]["indices"]
            for position in positions:
                factors[0, position, patch["head"], 0] = patch["factor"]
        modified = save("attention.head.modified", z * factors)
        projected = save("attention.projected", self.output(modified.reshape(b, t, d)))
        residual = save("attention.residual", replaced(x + projected, layer, "attention.residual", patches))
        mlp = save("mlp.output", self.mlpOut(F.gelu(self.mlpIn(self.ln2(residual)), approximate="tanh")))
        return save("block.output", residual + mlp)


class ReferenceTransformer(nn.Module):
    def __init__(self, c=Config()):
        super().__init__()
        self.config = c
        self.token = nn.Embedding(c.vocabularySize, c.modelDim)
        self.position = nn.Embedding(c.contextLength, c.modelDim)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.finalNorm = nn.LayerNorm(c.modelDim, eps=c.layerNormEpsilon)

    def forward(self, tokens, patches=(), capture_activations=True):
        capture = {} if capture_activations else None
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1]))
        if capture is not None:
            capture["embedding.output"] = x.detach().clone()
        for layer, block in enumerate(self.blocks):
            x = block(x, layer, patches, capture)
        normalized = self.finalNorm(x)
        if capture is not None:
            capture["final.normalized"] = normalized.detach().clone()
        logits = F.linear(normalized, self.token.weight)
        if capture is not None:
            capture["logits"] = logits.detach().clone()
        return logits, capture

    def exported_parameters(self):
        """Return canonical name -> (parameter, transpose on export)."""
        result = {"tokenEmbedding": (self.token.weight, False), "positionEmbedding": (self.position.weight, False)}
        def norm(prefix, module):
            result[prefix + ".scale"] = (module.weight, False)
            result[prefix + ".bias"] = (module.bias, False)
        norm("finalNorm", self.finalNorm)
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            norm(prefix + "ln1", block.ln1)
            norm(prefix + "ln2", block.ln2)
            for name in ("q", "k", "v", "output", "mlpIn", "mlpOut"):
                linear = getattr(block, name)
                result[prefix + name + ".weight"] = (linear.weight, True)
                result[prefix + name + ".bias"] = (linear.bias, False)
        return result


def serialized_tensor(value, transpose=False):
    if transpose:
        value = value.T
    value = value.detach().cpu().contiguous()
    return {"shape": list(value.shape), "values": value.flatten().tolist()}


def configuration(model):
    return asdict(model.config)
