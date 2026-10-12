# Computation inspection

Attention cells are selectable with a mouse or keyboard. Selecting a cell reads the baseline Q vector at the query position, K vector at the source position, raw score, and masked score row from the computation worker. The panel shows Q·K, head-dimension scaling, the actual computed score, the causal mask, stable-softmax numerator/denominator, and the attention weight. Expand Q/K products for individual dimensions.

The explanatory dot product and softmax use double-precision JavaScript arithmetic on read-back float32 vectors/scores. The worker's score and probability remain separately labeled; small rounding differences are expected. Future-position cells expose the -1e9 mask and zero probability. Same-layer head-output scaling occurs after attention weights are computed, so this map and arithmetic are unchanged by that operation.

The projection/residual/MLP panel is collapsed initially. Expanding it requests only the chosen stage and position for the completed baseline and modified runs. Its selectors can follow the contribution through the edited or a later layer:

1. Attention projection concatenates all head outputs and applies output weights plus the shared bias.
2. Attention residual adds the block input.
3. MLP output is the normalized residual's linear/GELU/linear contribution.
4. Block output adds that MLP contribution to the residual.

The panel defaults to the edited position for a single-position edit, or the prediction position for all-position edits. Each dimension shows baseline, modified, and difference. The table scrolls internally on narrow screens. An observation/reference key prevents a previous slice from appearing under newly selected layer/stage/position labels; obsolete requests cannot overwrite current results.

Exact strength entry validates finite values from 0 to 2, commits on Enter or blur, and leaves the completed experiment intact on invalid input. Slider and numeric entry share the same controls and worker execution path.

“Restart computation worker on CPU” terminates the old worker and creates a fresh session. It preserves valid requested settings against the completed input, and can cancel an unfinished load or probe. Queued computations belong to their original worker; their cleanup cannot unset the new worker's pending state. Existing run IDs, tensors, caches, and probe results are not reused in the new session.

Independent browser fixtures come from `scripts/export_inspection.py`, with seed-weight hashes checked against the installed verification artifact. They include the UI's BOS-prefixed six-token sequence, masked/unmasked cells, and all four stage vectors in both layers at factor .37. Production tests cover numerical parity, numeric validation, baseline reset, mobile containment, restart during edits, and cancellation of a blocked model load.

## Residual-stream read-out

**Read out the residual stream** (collapsed by default) shows the residual vector at the selected prediction position for the baseline and modified runs. It covers the token-plus-position embedding, each layer after attention, and each block output. Every row passes through the final LayerNorm and the tied unembedding, the operations that produce the model's logits (often called a "logit lens"). The panel shows each row's top token, the target's probability, and the residual's L2 norm. Only the last row is the model's prediction. Earlier rows show what that residual would predict if the computation stopped there; the model never computes them, and they are not claims about what a layer "knows."

The worker computes the read-out with one full forward pass, independent of cache state, using `TransformerModel.readOut`. The model's own logits use the same normalization and unembedding code. The panel reads nothing until it is opened.

Verification: `scripts/export_lens.py` loads the shipped retrieval weights into the independent PyTorch model. It checks that they reproduce both recorded logit vectors for the large-effect recipe, then exports read-outs for that recipe's baseline and ablated runs (`tests/fixtures/lens.json`; run `npm run fixtures:lens`). The engine matches all 120 read-out logits within the forward tolerance (maximum error 1.67e-6), and matches the norms. The final read-out row agrees with the model's logits within the same tolerance but is not bit-identical: a five-row matrix product sums float32 values in a different order than the 16-position forward pass. Invalid positions and released runs are rejected without retaining tensors. A production test compares every displayed top token, probability, and norm with the PyTorch fixture. It checks that the baseline and modified columns agree after reset, and that the panel scrolls within a 390 px viewport.

On that recipe, the baseline read-out first favors `3` after the layer-1 block (78.92%). With Layer 1 Head 4 ablated, the layer-1 block output still favors `3` (71.78%); the answer flips to `0` only after layer-2 attention. This illustrates downstream propagation for one input. It does not show which component "stores" the answer.

Adding this panel exposed a narrow-screen layout fault. On narrow screens the observation column was a `1fr` grid track, so a wide table could widen the whole column beyond the viewport, where the workspace clipped it. The column now has `min-width: 0`, so wide tables scroll inside their panels.
