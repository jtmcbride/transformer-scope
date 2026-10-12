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
