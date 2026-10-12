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

`residual.replace` overwrites one whole residual row with submitted values. Its fields are a zero-based `layer`, a `stage`, a `position`, and `values` of length `D`. Each value must be finite with magnitude at most 10,000.

- `block.input`: the row is replaced before block `layer`'s first LayerNorm, so that block's attention reads the replaced row. For layer 0 this happens after the token-plus-position embedding, and `embedding.output` itself is unchanged.
- `attention.residual`: the row is replaced after the projected attention output is added and before the MLP.

The captured stage reports exactly the submitted row, and all other rows are unchanged. Interaction with `head.scale` in the same layer follows execution order: a `block.input` replacement precedes that layer's attention, while an `attention.residual` replacement overwrites the head-scaled row at its position only. Two replacements at the same layer, stage, and position are rejected. A replacement is never treated as an identity, even when its values equal the run's own, because the planner cannot know that without computing.

A replacement at the prediction position reaches the prediction. A replacement at an earlier position reaches it if attention runs afterward: always for `block.input`, and only before the final layer for `attention.residual`. Runs containing a replacement use full execution. The cached suffix and cached inspections stay limited to same-layer head scaling, because cached tensors describe unreplaced residuals. Version 1 experiment files save head-scaling experiments only and refuse other kinds explicitly.

`attention.override` replaces one head's post-softmax attention row at one query position. Its fields are a zero-based `layer`, `head`, `query`, and `weights`, with one weight per input position.

- **Weights.** Each weight must be finite, non-negative, and at most 10,000. Every weight after `query` must be exactly 0, so the override stays causal, and the sum must be positive. The weights are divided by their sum in float64, then rounded to float32; the PyTorch reference does the same, so overridden rows are bit-identical.
- **What changes.** Scores and masked scores are unchanged. `attention.weights` reports the override, and `A × V` uses it, so the head's raw output changes at the query position only. Everything downstream follows.
- **Order within a layer.** A `block.input` replacement precedes the attention calculation. Head scaling applies to the head output computed from the overridden row. A row may be overridden once.
- **Reachability.** This follows the single-position head rule: the override reaches the prediction at the same position, or at a later one when a later layer exists.
- **Execution.** Overrides are always treated as effective, use full execution, and are inspected without cached tensors.

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

Baselines retain immutable descriptors and copied CPU logits. Modified descriptors copy the input and submitted interventions; they remain inspectable after the baseline is released. Inputs and request identity are copied before asynchronous work. Inspection returns axis names, shape, and a copied float32 slice, with a 65,536-value limit. Live runs are capped at 64; explicit release is idempotent.

The baseline cache retains raw head outputs `[1,T,H,D/H]` and attention residuals `[1,T,D]` for every layer, within a default 32 MiB logical float32 payload budget. Weights, temporary tensors, descriptor arrays, and driver allocations are outside that budget. LRU eviction skips pinned entries. A released entry with an active read lease is retired and disposed only after the last lease ends; if pinned entries consume all capacity, a new candidate is disposed and its run remains usable through full recomputation.

For effective edits in one layer, compute `Y' = Y + sum((factor - 1) * selected_Z * selected_W_output)`, with token masks, then execute that layer's MLP, subsequent blocks, final normalization, and unembedding. Output-projection bias cancels from the delta. Cross-layer edits and cache misses use full execution. Factor 1 and empty selections reuse baseline predictions. `executionPreference: "full"` forces the independent full path and remains in effect for that modified run's inspection.

Cache identity includes exact input tokens, model revision, architecture version, runtime generation, backend identity, dtype/precision settings, evaluation mode, and position/mask policy. Backend instance or precision-setting changes end the current session and dispose its caches. Weights are private to the engine; checkpoint loading creates a new session. Cached states cannot survive training updates or worker restart.

Inspection can read cached head/residual values, run a cached suffix for downstream values, or replay the full graph when the requested stage is not retained. Cache eviction does not erase run descriptors. Every returned temporary tensor is owned; helpers never return borrowed cache handles. Model disposal waits for active operations to finish, and all readback paths release tensors and leases in `finally` blocks. Non-finite baseline or intervened logits reject the request without committing a new run/cache entry.

## Training probe

The parity probe computes full-sequence mean cross-entropy, every parameter gradient, and one Adam update with learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, and epsilon `1e-5`.

Epsilon is explicit because key-projection bias gradients are analytically zero: adding a shared key bias shifts every score in a query row equally, leaving softmax invariant. Independent float32 reductions produce tiny residual gradients. An initial epsilon `1e-8` amplified those residual differences into approximately `1e-4` update disagreement. The conditioned `1e-5` probe preserves the original update comparison tolerances. It does not establish equivalence for every possible optimizer configuration.

The probe uses a separate model instance and cannot train or mutate the user's exploration model.

The first WebGL runtime disables tensor packing (`WEBGL_PACK=false`). Packed execution produced non-finite tied-embedding gradients on the tested WebKit runtime, while unpacked execution passed the same unchanged fixtures and tolerances. Treat packing as an optimization requiring separate graph parity before enabling it. The tiny verification model defaults to CPU to avoid cold GPU compilation overhead; WebGL remains explicitly selectable.
