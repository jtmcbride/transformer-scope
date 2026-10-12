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

The current engine suite passed 36 tests. Original forward/gradient results remain unchanged, measured using Node.js 24.18.0, TensorFlow.js Core/CPU 4.22.0, and the checked-in fixture:

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

## Cached suffix validation

- Additional independent fixture: three-layer models at seeds 1, 7, and 42; the same heads/dimensions/vocabulary/context as the original model; lengths 1, 5, and 8; early/middle/late layer edits; factors 0, 0.5, 1, 2; mixed heads and token selections. The Python exporter creates 261 cases containing weights, all-position logits, and attention residuals.
- Combining those cases with the original 84 yields 319 direct cached executions and 26 baseline/cross-layer cases. Direct factor-1 suffix execution is tested as well as the runtime identity shortcut.
- Maximum absolute cached-versus-full logit error: `1.430511474609375e-6`.
- Maximum absolute logit error against PyTorch across the combined full/cached checks: `2.384185791015625e-6`.
- Maximum absolute edited-residual error against PyTorch: `7.152557373046875e-7`.
- Comparisons retain the original forward bounds, `atol=2e-5`, `rtol=1e-5`; expected arrays and tolerances were not widened for caching.
- Tests verify full fallback after eviction/zero budget, LRU order, identity mismatch, rejected cache candidates, repeated reset, cached/full inspection across upstream and downstream stages, and immutable request snapshots.
- Controlled readback gates verify active lease pinning through release and eviction, deferred model disposal, and cleanup on readback failure. Backend changes and old checkpoint/runtime handles cannot reuse a cache.
- A deliberately overflowing finite-weight model rejects non-finite logits before admitting a baseline, with no retained tensor increase.
- The Node CPU stress test runs 1,000 comparisons with alternating layer/factor choices, reset cases, inspections, and invalid requests. Tensor counts remain at the loaded-model/cache baseline; releasing runs and disposing the engine returns to the initial count.

These checks do not establish GPU allocation-byte accuracy; TensorFlow.js logical tensor counts and backend estimates are distinct measurements.

## Browser verification

The production bundle includes a dedicated module worker and base-aware static model assets. Playwright tests run the built app at `/transformer-scope/` rather than a development-only module URL.

The browser suite checks:

- Two independent forward fixtures plus all gradients and an Adam update on CPU and, where supported, worker WebGL.
- Zero retained-tensor change after the probe disposes its separate model and optimizer.
- Rapid strength changes, reset, a causal no-effect control, and selective raw/modified vector inspection.
- Invalid input recovery and a 390-pixel viewport without horizontal overflow.
- Corrupted weight checksum rejection before computation.

The current production suite passed 26 tests across Chromium and WebKit. CPU and unpacked WebGL pass the original forward/gradient probe and the new cached/full suite in both engines. Each browser/backend tests 14 cached combinations, cross-layer fallback, independent PyTorch parity, 640-byte LRU eviction, modified-run replay, zero budget, and reload invalidation.

Each of the four browser/backend combinations also runs 1,000 sequential comparisons. Retained tensors remain at 40 (36 weights plus four cached tensors); releasing the baseline restores 36 tensors and zero logical cache bytes. The cache's 640-byte logical payload remains constant through the edits. Raw records are in `docs/cache-verification-results.json`; original probe records remain in `docs/browser-probe-results.json`. See `docs/benchmarks.md` for timing scope and the Firefox runtime launch limitation.

Playwright WebKit is not the installed Safari application, and viewport emulation is not physical mobile testing.

## Trained retrieval checkpoint

The independent trainer, dataset semantics, failed initial candidate, accepted checkpoint, split rule, and learning limitations are recorded in `docs/retrieval.md`. The manifest revision is `d95b28530c047010f403dc6b8c7256c7d9223df2cf83a02bd4a5a1bcb9d64eea`.

- Two Python tests check binding/target rules, all 3,072 inputs, disjoint splits, balanced targets for identical local contexts, length separation, and capture-free/reference gradient identity.
- Two training runs reproduced identical weights, tokenizer, and manifest hashes on the pinned CPU environment. Raw identity record: `docs/retrieval-reproduction-results.json`.
- The Node CPU engine reproduces all 453 independent held-out final-position logit fixtures and answers 453/453 correctly. Maximum absolute logit error: `8.203089237213135e-6`.
- Node trained forward-stage maximum error: `6.67572021484375e-6`; all-parameter gradient error: `3.5762786865234375e-7`; Adam update error: `1.2516975402832031e-6`. The probe uses full-sequence mean loss, while offline training supervises the final answer only.
- The three prepared recipes pass forced full and cached execution, all-vocabulary logit parity, target log-probability/JSD comparisons, reachability checks, and post-disposal tensor counts. Node maximum recipe logit error: `1.430511474609375e-6`.
- Browser CPU and unpacked WebGL pass the trained forward/gradient/Adam probe, all three recipes through full/cached paths, and resource cleanup in Chromium and WebKit. Raw numerical records: `docs/retrieval-browser-results.json`. Original tolerances remain unchanged.
- UI tests exercise recipe settings, removal of recipe claims after manual edits, target inference on custom valid bindings, checkpoint switching, mobile matrix scrolling, and page-width containment. Desktop/mobile screenshots were inspected.
- The shipped large-effect recipe ablates first-layer head 4 across all positions (indices are zero-based in JSON). Its correct-answer log-probability change is about -11.777 nats. The reachable control edits first-layer head 2 at position 0; its tiny effect is not treated as resolved beyond numerical error. The unreachable control edits the final layer at position 0 and leaves final-position logits exactly unchanged.

These finite-task checks demonstrate held-out sequence behavior and the selected experiments. They do not validate general language ability, average head importance, physical-device performance, or an untouched final audit dataset.

## Portable experiment validation

Version 1 export/replay passes three engine contract tests and four targeted production tests across Chromium and WebKit. They verify complete-vocabulary snapshots, non-final positions, target/scope restoration, checkpoint switching, CPU-to-WebGL replay, stale-checkpoint rejection before replacing the experiment, and visible discrepancies for self-consistent changed measurements. JSON size/version, input/token, intervention, numerical, and saved-metric consistency checks are covered. See `docs/experiment-format.md`.

Cold WebGL model loading is allowed 30 seconds in the replay browser assertions; this is a test startup allowance, not a warm performance claim.

## Computation-inspector validation

Independent UI fixtures check two attention cells (one causally masked) and the projection, attention residual, MLP, and block-output vectors in both layers for the actual BOS-prefixed screen input at factor .37. Production assertions use the existing 2e-5 absolute bound on displayed six-decimal values. Numeric strength rejection, reset to identical vectors, mobile containment, new-worker ownership during edits, and cancellation of an explicitly blocked model load pass in Chromium and WebKit. The two initially failing blocked-load checks were corrected to wait until the first request was actually intercepted before restarting; otherwise the test inadvertently blocked the replacement worker. No numerical tolerance was changed.

The final production regression passed all 26 tests across Chromium and WebKit after correcting the blocked-load test ordering. Engine regressions remain 36 passing tests. See `docs/computation-inspection.md`; mobile vector-panel screenshots were inspected.

## Local release verification

The final local checks passed: TypeScript/production build, 36 engine tests, 26 production browser tests, five Python dataset/reference/artifact tests, and source/built artifact audits. Standard-library dataset and artifact-corruption checks also pass without PyTorch installed in the active Python environment, matching the prepared CI commands.

`docs/browser-latency-results.json` records eight isolated named-host browser/backend/model benchmark runs, including cold-ready observations, 40 warm UI samples per layer, and 40 full/cached RPC pairs per layer. All measured frame-opportunity p95s are at most 68 ms on the recorded M1 Pro Mac. See `docs/benchmarks.md` for timer, paint, cache, capture, and device limitations.

The trained checkpoint's automatic CPU preference is tied to its measured revision. `?backend=cpu` and `?backend=webgl` permit explicit startup selection. CI and deployment inputs are documented in `docs/deployment.md`; local success does not itself confirm a remote CI run or a deployed website.

## Outstanding gates

- Wider trained-model intervention/seed coverage beyond the three shipped recipes.
- Physical Safari and mobile execution; GPU memory/context-loss recovery.
- Broader educational testing with users and additional controlled tasks.
- Named-device end-to-end latency benchmarks and actual hosted GitHub Pages verification.
- Larger-model, wider-seed, ill-conditioned-input, and lower-precision backend coverage.

Do not regenerate expected arrays or widen tolerances automatically after a failure. Inspect the earliest divergent tensor, artifact identity, matrix orientation, and backend precision first.
