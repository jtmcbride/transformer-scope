# Numerical contract: preln-absolute-tied-v1

This is the implemented contract shared by the independent PyTorch reference and TensorFlow.js engine. Changing it requires an architecture version change and reviewed fixture regeneration.

## Architecture

- Decoder-only, pre-LayerNorm, learned absolute position embeddings.
- Token and position embeddings are added at zero-based positions, including the beginning token.
- Each block computes `U = LN1(X)`, separate affine Q/K/V projections, causal attention, output projection, residual addition, and `MLP(LN2(residual))` followed by a second residual addition.
- A final LayerNorm precedes unembedding. Unembedding reuses the token-embedding matrix; there is no extra vocabulary bias.
- Every linear projection has bias. Dense weights are stored in `[input, output]` orientation.
- LayerNorm uses the final feature axis, population variance, epsilon `1e-5`, and affine scale and bias.
- MLP activation is `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
- No dropout, KV cache, padding, or sampling filters are used in this version.
- Inference uses batch size 1; all input positions are computed. A position's logits predict the token after that position.

## Attention and intervention

Projected Q/K/V and head outputs have shape `[B,T,H,D/H]`. Attention scores and probabilities have shape `[B,H,T,T]`, where rows are queries and columns are sources.

Scores are `Q K^T / sqrt(D/H)`. Future positions receive an additive `-1e9` mask before softmax. The supported verification fixtures produce zero future attention under this convention; arbitrary large activations are outside that guarantee and require separate validation.

`head.scale` multiplies the selected raw head output after `A × V` and before concatenation/output projection. It preserves that layer's attention weights. The output-projection bias and other heads are not scaled. All downstream operations use the modified residual.

Factors must be finite and within `[0,2]`. Layer, head, and token indices are zero-based. Overlapping patches and duplicate positions are rejected. Empty patch lists, empty selections, and factor 1 are identity operations. Forced full execution with factor 1 remains part of the parity tests.

## Tokenizer and metrics

The verification vocabulary is `<BOS>` followed by A–O. Input is split on whitespace without case or Unicode normalization; `<BOS>` is prepended automatically. Unknown symbols and excess context are errors. Empty input contains only `<BOS>`.

All metrics use the complete vocabulary at temperature 1. Display limits affect presentation only. Log-softmax and Jensen–Shannon divergence use natural logarithms (nats). Probability deltas are displayed in percentage points. Rank ties use ascending token ID.

## Artifacts

`manifest.json` describes all dimensions, conventions, tensor names, shapes, byte offsets, and exact checksums. `weights.bin` contains contiguous float32 little-endian values. `tokenizer.json` is UTF-8 JSON. No GPU tensors are serialized.

The exporter transposes native PyTorch `Linear.weight` tensors and their corresponding gradient/update fixtures. Tied weights are stored once. The runtime validates exact tensor membership/shapes, non-overlapping contiguous ranges, byte count, checksums, finite values, and supported numerical conventions before allocating model tensors.

The model revision is the SHA-256 of the canonical manifest before adding the revision field. Weight/tokenizer checksums are verified on load. The revision is an artifact identifier rather than a signature from a trusted publisher.

Current loader limits: 16 layers, model dimension 512, MLP dimension 2048, vocabulary 8192, context 512, and 64 MiB weight payload. These are allocation guards, not demonstrated interactive capabilities.

## Ownership and runs

The worker owns weights and serializes model loading, execution, and release. Every runtime gets a distinct generation ID. Loading another model clears prior run handles.

Baselines retain immutable descriptors and copied CPU logits. Modified descriptors copy the input and submitted interventions; they remain inspectable after the baseline is released. Intermediate activations are recomputed selectively and disposed after readback. Inspection returns axis names, shape, and a copied float32 slice, with a 65,536-value limit. Live runs are capped at 64; explicit release is idempotent.

There is no suffix cache yet. Effective interventions use full execution. Identity comparisons reuse retained baseline predictions. Cached suffix execution is the next optimization milestone.

## Training probe

The parity probe computes full-sequence mean cross-entropy, every parameter gradient, and one Adam update with learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, and epsilon `1e-5`.

Epsilon is explicit because key-projection bias gradients are analytically zero: adding a shared key bias shifts every score in a query row equally, leaving softmax invariant. Independent float32 reductions produce tiny residual gradients. An initial epsilon `1e-8` amplified those residual differences into approximately `1e-4` update disagreement. The conditioned `1e-5` probe preserves the original update comparison tolerances. It does not establish equivalence for every possible optimizer configuration.

The probe uses a separate model instance and cannot train or mutate the user's exploration model.

The first WebGL runtime disables tensor packing (`WEBGL_PACK=false`). Packed execution produced non-finite tied-embedding gradients on the tested WebKit runtime, while unpacked execution passed the same unchanged fixtures and tolerances. Treat packing as an optimization requiring separate graph parity before enabling it. The tiny verification model defaults to CPU to avoid cold GPU compilation overhead; WebGL remains explicitly selectable.
