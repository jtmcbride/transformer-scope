# Validation record

These results describe the first implementation, not a completed product release.

## Reference provenance

- Independent Python/PyTorch implementation: `scripts/reference_model.py`.
- Exporter: `scripts/export_fixtures.py`, generator version 1, seed 1729, PyTorch 2.9.1 on CPU, one PyTorch thread.
- Actual arrays are exported; JavaScript does not recreate weights from a matching seed.
- Model: 2 layers, 2 heads, embedding dimension 8, head dimension 4, MLP dimension 16, vocabulary 16, context limit 8.
- Artifact: 36 weight tensors, 1,408 float32 parameters, 5,632 weight bytes. Token embedding and unembedding are tied.
- Fixture set: 84 cases at lengths 1, 2, 5, 8; both layers and heads; factors 0, 0.5, 1, 2; all/single/disjoint token selections; same-layer and cross-layer edits.
- Every exported case captures embeddings, block inputs, attention normalization, Q/K/V, scores, masked scores, weights, raw/modified head outputs, projected output, residuals, MLP, block outputs, final normalization, and all-position logits.
- Training fixture: input `[1,5,9,3,7]`, targets `[5,9,3,7,2]`, mean cross-entropy over all positions, all parameter gradients, one Adam update.

## CPU numerical results

The final engine suite passed 13 tests. Measured using Node.js 24.18.0, TensorFlow.js Core/CPU 4.22.0, and the checked-in fixture:

- Maximum absolute forward-stage error across all 84 cases: `1.9073486328125e-6`.
- Maximum absolute gradient error across every parameter: `3.5762786865234375e-7`.
- Maximum absolute Adam update error across every parameter: `5.587935447692871e-7`.

The assertions use `abs(actual - expected) <= atol + rtol * abs(expected)`. Forward/loss: `atol=2e-5`, `rtol=1e-5`; gradients: `atol=3e-5`, `rtol=3e-4`; updated weights: `atol=3e-5`, `rtol=1e-5`. These are initial regression bounds for the specified fixtures, not universal accuracy guarantees.

Central finite differences check four selected entries spanning embeddings, attention, MLP, and final normalization. Perturbation is `0.001`; diagnostic bounds are `atol=7e-4`, `rtol=3e-3`. This limited sample complements all-parameter independent gradient parity; it does not exhaustively finite-difference every parameter.

The optimizer probe uses Adam epsilon `1e-5`. The initial `1e-8` probe exposed approximately `1e-4` update divergence on analytically zero key-bias gradients because Adam amplified float32 reduction roundoff. See the model contract for the reason and scope of the conditioned probe. Assertions were not widened to accommodate that discrepancy.

## Behavioral and resource checks

- Forced full factor-1 execution is included in the independent fixtures.
- Later-position edits cannot change earlier logits.
- Earlier-position final-layer head-output edits cannot change the final-position prediction.
- Modified runs retain their original input/interventions when callers change submitted arrays or objects.
- Modified activation inspection works after explicit release of the baseline descriptor.
- Identity comparisons reuse copied baseline logits; no forward recomputation is needed.
- Invalid tokens, overlapping interventions, duplicate positions, non-finite factors, incompatible tensor shapes, and invalid inspection axes/indices are rejected.
- Fifty repeated full-forward capture/disposal cycles, interleaved with rejected interventions, retain a constant tensor count. Disposing the model returns to the pre-load count.
- The live-descriptor limit and failed-comparison cleanup are tested; release and disposal are idempotent.

The planned 1,000-intervention suffix-cache stress test is deferred until that cache exists. These checks do not establish GPU allocation-byte accuracy; TensorFlow.js logical tensor counts and backend estimates are distinct measurements.

## Browser verification

The production bundle includes a dedicated module worker and base-aware static model assets. Playwright tests run the built app at `/transformer-scope/` rather than a development-only module URL.

The browser suite checks:

- Two independent forward fixtures plus all gradients and an Adam update on CPU and, where supported, worker WebGL.
- Zero retained-tensor change after the probe disposes its separate model and optimizer.
- Rapid strength changes, reset, a causal no-effect control, and selective raw/modified vector inspection.
- Invalid input recovery and a 390-pixel viewport without horizontal overflow.
- Corrupted weight checksum rejection before computation.

The final production suite passed eight tests across Chromium and WebKit. CPU and unpacked WebGL passed the numerical probe in both engines. Probe results include user agent, active backend, float32 texture status, packing flag, numerical errors, and elapsed time; a copy is retained in `docs/browser-probe-results.json`. See `docs/benchmarks.md` for exact measured errors and the Firefox runtime launch limitation.

Playwright WebKit is not the installed Safari application, and viewport emulation is not physical mobile testing.

## Outstanding gates

- Cached suffix equivalence, eviction, and 1,000-edit resource stress.
- Physical Safari and mobile execution; GPU memory/context-loss recovery.
- Controlled-task learning, held-out behavior, and measured educational recipes.
- Named-device end-to-end latency benchmarks and actual hosted GitHub Pages verification.
- Larger-model, wider-seed, ill-conditioned-input, and lower-precision backend coverage.

Do not regenerate expected arrays or widen tolerances automatically after a failure. Inspect the earliest divergent tensor, artifact identity, matrix orientation, and backend precision first.
