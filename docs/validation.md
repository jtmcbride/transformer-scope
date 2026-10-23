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

The current production suite passed 28 tests across Chromium and WebKit. CPU and unpacked WebGL pass the original forward/gradient probe and the new cached/full suite in both engines. Each browser/backend tests 14 cached combinations, cross-layer fallback, independent PyTorch parity, 640-byte LRU eviction, modified-run replay, zero budget, and reload invalidation.

Each of the four browser/backend combinations also runs 1,000 sequential comparisons. Retained tensors remain at 40 (36 weights plus four cached tensors); releasing the baseline restores 36 tensors and zero logical cache bytes. The cache's 640-byte logical payload remains constant through the edits. Raw records are in `docs/cache-verification-results.json`; original probe records remain in `docs/browser-probe-results.json`. See `docs/benchmarks.md` for timing scope and the Firefox runtime launch limitation.

Playwright WebKit is not the installed Safari application, and viewport emulation is not physical mobile testing.

## Trained retrieval checkpoint

The independent trainer, dataset semantics, failed initial candidate, accepted checkpoint, split rule, and learning limitations are recorded in `docs/retrieval.md`. The manifest revision is `d95b28530c047010f403dc6b8c7256c7d9223df2cf83a02bd4a5a1bcb9d64eea`.

- Two dataset tests and one reference test check binding/target rules, all 3,072 inputs, disjoint splits, balanced targets for identical local contexts, length separation, and capture-free/reference gradient identity.
- Two training runs reproduced identical weights, tokenizer, and manifest hashes on the pinned CPU environment. Raw identity record: `docs/retrieval-reproduction-results.json`.
- The Node CPU engine reproduces all 453 independent held-out final-position logit fixtures and answers 453/453 correctly. Maximum absolute logit error: `8.203089237213135e-6`.
- Node trained forward-stage maximum error: `6.67572021484375e-6`; all-parameter gradient error: `3.5762786865234375e-7`; Adam update error: `1.2516975402832031e-6`. The probe uses full-sequence mean loss, while offline training supervises the final answer only.
- The three prepared recipes pass forced full and cached execution, all-vocabulary logit parity, target log-probability/JSD comparisons, reachability checks, and post-disposal tensor counts. Node maximum recipe logit error: `1.430511474609375e-6`.
- Browser CPU and unpacked WebGL pass the trained forward/gradient/Adam probe, all three recipes through full/cached paths, and resource cleanup in Chromium and WebKit. Raw numerical records: `docs/retrieval-browser-results.json`. Original tolerances remain unchanged.
- UI tests exercise recipe settings, removal of recipe claims after manual edits, target inference on custom valid bindings, checkpoint switching, mobile matrix scrolling, and page-width containment. Desktop/mobile screenshots were inspected.
- The shipped large-effect recipe ablates first-layer head 4 across all positions (indices are zero-based in JSON). Its correct-answer log-probability change is about -11.777 nats. The reachable control edits first-layer head 2 at position 0; its tiny effect is not treated as resolved beyond numerical error. The unreachable control edits the final layer at position 0 and leaves final-position logits exactly unchanged.

These finite-task checks demonstrate held-out sequence behavior and the selected experiments. They do not validate general language ability, average head importance, physical-device performance, or an untouched final audit dataset.

## Portable experiment validation

Exports are now schema version 2 (`docs/experiment-format.md`). Three engine tests cover version 1 compatibility, redirect round trips and refusals, and patch round trips with source-input checks. Production tests export, reset, and import an attention redirect and a residual patch. They require the restored mode and controls and a matching replay, including a patch imported while the other checkpoint was loaded. The version 1 results below still apply to head scaling.


Version 1 export/replay passes three engine contract tests and four targeted production tests across Chromium and WebKit. They verify complete-vocabulary snapshots, non-final positions, target/scope restoration, checkpoint switching, CPU-to-WebGL replay, stale-checkpoint rejection before replacing the experiment, and visible discrepancies for self-consistent changed measurements. JSON size/version, input/token, intervention, numerical, and saved-metric consistency checks are covered. See `docs/experiment-format.md`.

Cold WebGL model loading is allowed 30 seconds in the replay browser assertions; this is a test startup allowance, not a warm performance claim.

## Computation-inspector validation

Independent UI fixtures check two attention cells (one causally masked) and the projection, attention residual, MLP, and block-output vectors in both layers for the actual BOS-prefixed screen input at factor .37. Production assertions use the existing 2e-5 absolute bound on displayed six-decimal values. Numeric strength rejection, reset to identical vectors, mobile containment, new-worker ownership during edits, and cancellation of an explicitly blocked model load pass in Chromium and WebKit. The two initially failing blocked-load checks were corrected to wait until the first request was actually intercepted before restarting; otherwise the test inadvertently blocked the replacement worker. No numerical tolerance was changed.

The final production regression passed all 28 tests across Chromium and WebKit after correcting the blocked-load test ordering. Engine regressions remain 36 passing tests. See `docs/computation-inspection.md`; mobile vector-panel screenshots were inspected.

## Local release verification

The final local checks passed: TypeScript/production build, 36 engine tests, 28 production browser tests, five Python dataset/reference/artifact tests, and source/built artifact audits. Standard-library dataset and artifact-corruption checks also pass without PyTorch installed in the active Python environment, matching the prepared CI commands.

`docs/browser-latency-results.json` records eight isolated named-host browser/backend/model benchmark runs, including cold-ready observations, 40 warm UI samples per layer, and 40 full/cached RPC pairs per layer. All measured frame-opportunity p95s are at most 68 ms on the recorded M1 Pro Mac. See `docs/benchmarks.md` for timer, paint, cache, capture, and device limitations.

The trained checkpoint's automatic CPU preference is tied to its measured revision. `?backend=cpu` and `?backend=webgl` permit explicit startup selection. CI and deployment inputs are documented in `docs/deployment.md`; local success does not itself confirm a remote CI run or a deployed website.

## Remote CI

The first GitHub Actions run passed all configured checks on Ubuntu 24.04: locked dependency install, four standard-library dataset/artifact tests, 36 engine tests, production build, source/built audits, and 13 Chromium production tests. Commit: `00e814e`; [workflow run](https://github.com/jtmcbride/transformer-scope/actions/runs/38254063282). The static build and browser diagnostics were uploaded successfully. This verifies CI, not hosted-site publication.

The extended Chromium/Firefox run passed all 28 production tests at `6a41a09`: [run 38254919942](https://github.com/jtmcbride/transformer-scope/actions/runs/38254919942). On Ubuntu 24.04, Chromium 156 passed CPU/unpacked WebGL, and Firefox 157 passed CPU; Firefox worker WebGL was explicitly unsupported. No test was skipped or flaky. Exact user agents, numerical probes, resource results, and test annotations are in `docs/ci-browser-results.json`. Browser checks record an explicitly unsupported WebGL load separately and continue validating CPU behavior. Numerical failures after a supported load are still failures. Worker WebGL now requires float32 rendering support; lower-precision contexts are rejected with `UNSUPPORTED_BACKEND` and can recover on CPU. A capability test emulates missing worker WebGL context support while running real CPU inference and recovery.

## Hosted release and native Safari

Commit `71a5694` passed the full verification/deployment pipeline on Ubuntu 24.04: four standard-library dataset/artifact checks, 36 engine tests, 28 Chromium/Firefox production cases, Pages deployment, and two hosted Chromium cases. [Run 38258414801](https://github.com/jtmcbride/transformer-scope/actions/runs/38258414801). The live URL is https://jtmcbride.github.io/transformer-scope/.

Hosted checks verify the exact clean source commit and all 14 deployment asset hashes, byte lengths, MIME types, and model revisions. The live worker also passes prepared recipes, reference verification, reset, export/import replay, lazy vector inspection, CPU restart, and checkpoint switching without page errors or failed requests. The same two tests passed from the local Mac against the actual HTTPS host. `docs/hosted-release-results.json` preserves evidence.

Installed Safari 27.0.1 (22625.1.29.11.28) on the recorded M1 Pro / macOS 27.0.1 host was exercised through native UI controls. Local production CPU and unpacked WebGL both passed the PyTorch probe: maximum displayed forward error 6.68e-6, gradient error 3.58e-7, Adam error 1.25e-6 CPU / 2.08e-6 WebGL, retained tensor change zero. Baseline reset, lazy computation vectors, and explicit GPU-to-CPU restart passed. The hosted HTTPS site loaded the prepared ablation and passed the CPU reference probe. `docs/native-safari-results.json` records build provenance and the scope of these manual checks; this is not a full installed-Safari automated regression or a Safari timing benchmark.

## Device validation harness

`validation.html` is an opt-in diagnostic page at the deployment root (https://jtmcbride.github.io/transformer-scope/validation.html). The application does not link to or load it, and it runs nothing until **Start validation** is pressed. For each selected checkpoint and backend, it records:

- Release identity (`release.json` commit and model revisions), browser/OS reports, touch and viewport properties, timer resolution, and the main-thread WebGL renderer. The observer enters device model, power, and thermal state, and chooses an evidence class; `navigator.webdriver` is recorded separately.
- In a fresh worker: actual backend and precision, the PyTorch forward/gradient/Adam probe, all three recipes through cached and full execution with recorded expectations, in-memory export/replay, and post-release tensor/cache counts. These use the existing tolerances (`2e-5 + 1e-5|expected|` for logits).
- In the real application, embedded from the same deployment: cold readiness, warm input-to-result timings, baseline reset, and CPU worker recovery.

Each check is `passed`, `failed` (with the reason), `unsupported` (the backend could not start), or `skipped` (not applicable or after an unsupported backend). An unsupported CPU fails the run. Timing definitions are in `docs/benchmarks.md`; the run also records whether the page was hidden, which invalidates its timings.

Coverage: five engine tests check timing summaries, the logit bound, recipe expectations against the recorded PyTorch metrics and causal reachability, settings, and status aggregation. Two production tests run in Chromium and Firefox. The first confirms that the application loads no harness code and that the harness requests no worker, model, or release asset before Start. It then runs both checkpoints and backends, verifies identity, every check, separated timings, and an identical JSON download. The second emulates missing worker WebGL and corrupts one PyTorch probe value. It verifies that the probe fails with its mismatch, WebGL is reported as unsupported with dependent checks skipped, independent CPU checks still pass, and the page fits a 390-pixel viewport. A hosted test runs the harness against the deployed site.

At commit `29b6ade`, CI passed all 32 production cases (16 each in Chromium 156 and Firefox 157, including both harness tests): [run 38263433282](https://github.com/jtmcbride/transformer-scope/actions/runs/38263433282). The deployment job then published that commit, and all three hosted Chromium checks passed against https://jtmcbride.github.io/transformer-scope/, including a harness run that identified commit `29b6ade` and had no failed checks. In this development container, headless Chromium 141 on Linux also passed all 16 Chromium cases and the three hosted-suite tests against a local production preview. These are automation results, not device evidence. A full-default run (both checkpoints and backends, 5 warm-up and 40 measured edits per layer) against a clean local build of `29b6ade` passed every applicable check; its unedited export is `docs/device-results/2026-10-12-automation-container-headless-chromium141.json`, kept as a format reference and labeled as automation.

## WebGL context loss

Mobile browsers can reclaim a backgrounded page's GPU context. Before this change, a lost worker context left the application on "Computing…" indefinitely, with no error. The test reproduced this in production Chromium by dropping the worker's context through `WEBGL_lose_context`. TF.js keeps using the lost context, so readbacks stall; on other drivers they could return meaningless values.

The worker now listens for `webglcontextlost` and checks `isContextLost()` before and after every request. On loss, it discards the request's result, replies with `STALE_SESSION`, and posts a notice that ends the client session. Pending and later requests then reject with the same reason. The application shows "The browser reclaimed the GPU (WebGL context lost)…" with **Restart on CPU**, including when the loss happens while idle. Results completed before the loss stay on screen. The worker never restores a lost context, and it never switches backend silently.

Two production tests inject a browser-initiated loss into the real worker bundle, once while idle and once just after a comparison starts. Both require the alert, no display of the requested post-loss result, and recovery of the experiment on CPU. With the previous worker, both fail by timing out. Firefox CI has no worker WebGL, so these tests annotate it as unsupported there. The harness adds a `webgl-context-loss` check that runs the same path on any device. In local Chromium, the session ended 10–14 ms after the loss, and the replacement CPU worker passed the PyTorch probe.

## Touch emulation

At 360 CSS px in Chromium with touch emulation, the strength slider's hit area was 16 px tall and the file input 17 px, below the 24 px WCAG 2.2 minimum target size. On coarse pointers, both are now 32 px; the mouse layout is unchanged. A production test checks every control except inline text links for at least 24 × 24 px, checks containment, and sets the strength with a touchscreen tap. It fails without the CSS change. If an engine does not report a coarse pointer under emulation, the test annotates that and asserts containment only. This is viewport/touch emulation, not physical-device evidence.

## Ablation explorer

The per-prediction head-ablation scan matches independent PyTorch logits for three inputs, covering all single heads and three pairs (maximum error 4.29e-6, unchanged bounds). Its metrics are identical to the shared comparison code applied to its logits. Details and the measured non-additive pairs: `docs/computation-inspection.md`.

## Attention overrides

`scripts/export_attention.py` (`npm run fixtures:attention`) loads the shipped weights into the independent PyTorch model and writes `tests/fixtures/attention-reference.json`. It exports five verification-model cases:

- one-hot and uniform rows, including unnormalized weights;
- a final-layer override at an earlier query, which cannot reach the prediction;
- an override combined with head scaling and a `block.input` replacement in the same layer.

It also exports a 64-case retrieval routing sweep: each final-layer head at query 15 of `A = 0 ; B = 3 ; C = 0 ; ? B =`, sent entirely to each source position. Engine results:

- Logits and captured scores, weights, raw head outputs, attention residuals, and block outputs match within the unchanged forward bounds (maximum error 1.43e-6). Overridden rows equal both the engine's normalization and the PyTorch rows bit for bit.
- Scores are unchanged unless an earlier edit changed the layer's input. The unreachable case leaves the prediction's logits bit-identical.
- The run engine plans full execution, reports reachability, and inspects overridden weights without cached tensors.
- The routing sweep matches within the bounds. Its largest absolute error is 1.85e-5, where redirected rows push logits to magnitudes around 15, so the relative term applies.
- Invalid lengths, later-position weight, negative or non-finite values, a zero sum, out-of-range indices, and duplicate rows are rejected without retaining tensors.

**Redirecting attention on screen.** **Intervention type → Redirect a head's attention** takes a layer, head, query position, and target: all weight on one allowed source, or evenly over every allowed position. The attention inspector shows the modified run's weights, with the replaced row identified. The head-output panel shows that head's output at the query before and after. Cell arithmetic is hidden in this mode, because the override bypasses the softmax it explains, and export stays disabled. A production test compares displayed probabilities with the PyTorch sweep for four targets, and checks the displayed row, focus clamping when the query moves, the no-path note, and reset.

For this input, Layer 2 Head 2 at query 15 puts nearly all of its weight (1.000) on position 14, the queried key. Sending it to position 3 or 5 makes `0` the prediction (73% and 100%); the other final-layer heads change little when redirected. This is routing measured on one input, not a head role.

## Residual replacement

`scripts/export_residual.py` loads the shipped weights into the independent PyTorch model; seeded re-initialization is not bit-reproducible across PyTorch builds. It exports seven verification-model cases and a 64-case retrieval patching sweep (`tests/fixtures/residual-reference.json`; `npm run fixtures:residual`). The cases cover both hooks in both layers, first, middle, and last positions, a final-layer replacement that cannot reach the prediction, a zero vector, and a combination with head scaling.

The sweep uses `A = 0 ; B = 3 ; C = 0 ; ? B =` as the target and the same input with the query key `A` as the source; the two differ only at position 14. It replaces each position's target residual with the source residual at each hook in each layer.

Engine results:

- Full execution matches every logit and every captured `block.input`, `attention.residual`, and `block.output` within the unchanged forward bounds (maximum error 1.91e-6). Each hook reports exactly the submitted row.
- Replacing a row with the engine's own value reproduces the baseline logits bit for bit. The final-layer replacement at an earlier position leaves the prediction's logits bit-identical.
- The run engine plans full execution, reports reachability, inspects replaced runs without cached tensors, and replays them after the baseline is released.
- The sweep, with source residuals read through `inspectActivation`, matches PyTorch within 2.44e-6.
- Invalid lengths, non-finite or out-of-range values, unknown stages, out-of-range positions or layers, and duplicate targets are rejected without retaining tensors.

Regenerating the existing head-scaling fixtures with and without the reference change produced byte-identical files on this host. Those regenerations used PyTorch 2.9.1+cu128 on Linux and differ from the committed macOS CPU-build fixtures, because seeded initialization differs across builds. The committed fixtures were left unchanged.

**Patching on screen.** **Intervention type → Patch the residual from another input** takes a source input with the same number of symbols. For a well-formed retrieval input it suggests the same bindings queried for another key. You choose a patch point (a layer's input or its post-attention residual) and a position; positions that differ between the inputs are marked ←. The worker reads the source run's residual at that point and submits it as `residual.replace` on the current input. The prediction comparison, residual read-out, embedding, and computation panels then show the effect, and a patched-residual panel lists the replaced row.

Source runs are released when the input, model, or worker changes. A CPU worker restart recreates the source run and keeps the patch settings. Changing the input returns to head scaling, and export stays disabled because version 1 files save head scaling only. A production test compares the displayed probabilities of `3` and `0` with the PyTorch sweep at six patch points, using the original two-decimal display bound. It also checks the no-path note for the final-layer earlier-position case, full execution, the 64-dimension patched row, restart, source-length validation, and reset.

In the sweep, the source's query key moves the prediction from `3` to `0` (target probability 99.99% to about 0%) only through position 14 up to layer 2's input, and only through position 15 after layer 2's attention. This holds for one input pair; it is not a general claim about where the model stores information.

## Residual-stream read-out

The input-embedding panel matches independent PyTorch embedding rows, sums, and similarity rows at four positions; its sum is bit-identical to the model's captured embedding output. The logit-lens panel matches independent PyTorch read-outs for the large-effect recipe's baseline and ablated runs. The maximum logit error is 1.67e-6 under the unchanged forward bounds (2e-5 absolute, 1e-5 relative), and the norms also match. Details and limitations: `docs/computation-inspection.md`.

## Physical-device results

None. On 2026-10-12, the project owner chose to skip the lower-powered phone run and proceed to educational sessions. The physical-mobile gate is deferred, not passed. No physical device has run the harness or the checklist, and there is no physical-mobile performance or compatibility claim. To close the gate later, run `docs/physical-device-protocol.md`. Run `docs/physical-device-protocol.md` and record the harness JSON and manual checklist in `docs/device-results/`. Do not substitute viewport emulation, Playwright WebKit, installed desktop Safari, or automated harness runs for this gate.

## Educational evaluation

`docs/educational-evaluation.md` defines a facilitated session around the three recipes: participants predict and explain each result before seeing it, then answer probes on attention-weight invariance under head-output scaling, downstream propagation, small effects versus no path, bounded interpretation, and a reachability transfer question. Gate criteria are fixed before data collection. `docs/educational-findings-template.md` is the findings template. The opt-in session page `study.html` follows the protocol: it locks each prediction before loading the live recipe result in the real application, reads the displayed result, and exports a per-participant JSON record. `scripts/summarize_study.mjs` validates the records and produces the findings tables, but leaves the gate unevaluated until people assign the probe codes.

Coverage: four engine tests check that the prediction cards match `recipes.json` in UI numbering, the observed-category rules (including the sign of a rounded `-0.0000` change), record validation (ID format, consent, task order, prediction before reveal), and the summary's gate arithmetic. They also check that a correct answer after a hint does not count, and that missing codes leave the gate unevaluated. Two production tests check the following. The application loads no session code. Each task has no embedded application until its prediction is locked, and then shows the requested recipe. The three recorded categories are down/large, down/tiny, and same/none with the no-path note. The download matches the record, contains no rejected free-text ID, and the page fits 390 px. Finally, an unfinished session resumes after a reload and can be discarded. No participant sessions have been run, and no participant results exist.

## Outstanding gates

- Wider trained-model coverage: single-head ablations across all 453 held-out inputs are recorded (`docs/retrieval.md`, "Head-ablation coverage"). Other factors, prediction positions, head interactions, additional training seeds, and independent PyTorch recomputation of those comparisons remain open.
- A physical lower-powered mobile device: run `docs/physical-device-protocol.md` (harness plus manual checklist). Context-loss detection is tested with simulated loss in Chromium and by the harness; real backgrounding on mobile GPUs is covered only by step B12. GPU memory exhaustion is not tested.
- Educational sessions under `docs/educational-evaluation.md` (at least five participants), then wider tasks if the gate is met.
- Broader named-device end-to-end latency benchmarks, including native Safari and physical mobile; current measurements cover one desktop host.
- Larger-model, wider-seed, ill-conditioned-input, and lower-precision backend coverage.

Do not regenerate expected arrays or widen tolerances automatically after a failure. Inspect the earliest divergent tensor, artifact identity, matrix orientation, and backend precision first.

## TinyStories real-text release

The shipped 48,000-step checkpoint adds two engine tests (74 total): exact text tokenization/export/replay and independent Python baseline/full/suffix logits for three prompts. Production worker checks pass the complete forward, gradient, Adam, and intervention fixtures on local Chromium and WebKit, using CPU and supported float32 WebGL. Each case and numerical probe retains zero extra tensors. Exact reports and the scale-aware error bounds are recorded in `tinystories-numerical-results.json` and `tinystories-deployment.md`.

The final compact layer navigation and real-text workflow pass 12 focused Chromium/WebKit cases. These cover navigation without changing the prediction, head edits, model switching, live replay, exact Unicode and whitespace export, worker restart preserving the completed prompt, context overflow retaining the last result, and the existing visual flow controls. WebKit can compile a new GPU shape slowly; text workflows wait for actual completed status for up to 60 seconds, without changing numerical tolerances. The complete release suite has 55 browser cases per browser and five hosted checks, including the default English prompt and every exported next-token logit.

The original name-copying gate remains failed (56%, required 80%). Full-run training determinism, the untouched language-quality audit, and physical mobile performance are not established by these checks. The diagram predicts one next token with a 64-token context including BOS.

## Controlled continuation

Four additional engine cases (78 total) cover sampling/stop policies, atomic paired failures, exact UTF-8 prefixes, and a 16-step original/edited rollout against independent shipped-weight PyTorch outputs, with no retained tensors. Three additional production cases per browser cover paired stepping/playback, frozen edit snapshots, seeded replay, completed-versus-draft prompts, context exhaustion, mobile containment, and cancellation during deliberately delayed worker inference. The hosted suite has a sixth check for the same original/edited rollout and its downloaded record. See `controlled-continuation.md` for the exact sampling, runtime, and resource contracts. Generation is bounded to this model's existing context; there is no claim of complete-story quality or physical-mobile generation speed.

## Visual flow redesign

The default visual diagram and the full Detailed workspace share the completed experiment state. New Chromium/WebKit checks cover keyboard node selection, head edits, export after view switching, baseline reset, mobile containment, bounded canvas scrolling, and draft input separation. Existing workflow regressions explicitly open `?view=details` to keep exercising all numerical and advanced controls. The build and artifact audits pass.

Touch emulation exposed WebKit native selects ignoring minimum sizing; the coarse-pointer style now draws a select indicator and supplies a usable control height. The opt-in validation harness now scrolls its iframe into view before timing edits, because WebKit throttles offscreen animation frames. Its 30-second timeout also covers the two animation callbacks, and exported timing records identify the visual workspace. Both harness tests pass in Chromium and WebKit after this fix. These remain automated browser checks, not physical-device evidence.

The follow-up puts head presets, a continuous strength slider, and baseline restoration inside the diagram. Eighteen focused production cases pass across Chromium/WebKit, covering both workspace views and touch layout. A delayed worker check confirms that rapid Normal/Amplify/Mute edits preserve the last completed result and that an exact factor at a selected position exports correctly after switching views. Hosted verification also exercises the default flow against the deployed worker and confirms the prepared answer changes from 3 to 0 under ablation and returns to 3 on reset.


## Generated-token inspection

The continuation suite now includes exact immutable historical-prefix selection (79 engine tests total). Production browser checks inspect both branches after divergence, compare selected-token probabilities to the independent PyTorch rollout, verify a muted head's zero output and full vector dimensions, navigate layers/tokens, replay sampled choices, preserve downloaded stories, and handle keyboard/mobile navigation. A new delayed-worker case exercises token switching, close, reset, pause-by-selection, and model changes without stale results. There are 59 production cases per browser. The existing sixth hosted check additionally verifies edited step seven's prefix, probability, and head output on the deployed build. These checks do not establish physical-mobile performance or broader language quality.


## Historical story branching

Three additional engine tests (82 total) cover sampling-counter continuity, exact prefix/history preservation, inherited/replaced edits, first-prediction expiry, atomic failures, EOS/context limits, and four independent 12-token PyTorch branch pairs with no retained tensors. Three production browser cases per browser (62 total per browser) cover those pairs plus frozen results, unchanged source records, mobile containment, seeded identity forks, cancellation/resume, and source-token navigation during delayed inference. The sixth hosted check also runs and exports a branch from edited step seven, replacing its captured head edit. Earlier numerical tolerances and model-quality gates are unchanged.


### Nested story inspection and branching

Two additional engine tests (84 total) verify selecting exact prefixes from either fork, global seeded draw counters, multi-head inheritance, first-prediction expiry across generations of comparisons, and three independent nested PyTorch rollouts without retained tensors. Two additional production cases (64 per browser) cover both paths, two levels of further edits, preserved ancestor records, keyboard navigation/focus, narrow-screen layout, and bounded worker counts. The existing delayed-worker case now also checks sampled branch-token replay and returning while a deeper comparison is in flight. The sixth hosted check inspects a branch result and generates another paired continuation. Local and hosted execution results are reported separately; numerical tolerances, context length, and checkpoint quality gates are unchanged.


### Persistent story workspace

A new engine test (85 total) covers whole-tree round trips, parent-source validation, checkpoint mismatch, invalid links/probabilities/routes, and exact paused-fork resume. The existing deep-branch browser case now creates three comparisons, navigates between them, reloads, exports/imports the full workspace, and refuses invalid input without losing results. Two new browser cases (66 per browser) verify seeded root resume after reload and visible storage failure with downloadable state. The hosted continuation case now reloads the deployed workspace and reopens its nested comparison through the map. These checks preserve earlier numerical tolerances and checkpoint quality boundaries; they do not establish cross-tab synchronization or cloud backup.


### Branch-point probability explanations

Two new engine cases (87 total) verify raw softmax versus generation eligibility, stable ranks/ties, selected-token inclusion, seeded choices, identity edits, and seven independent same-prefix PyTorch comparisons with zero retained tensors. `scripts/export_branch_probabilities.py` exports top alternatives, ranks, and raw probability changes for the original, inherited, replaced, nested, and temporary head-edit cases. Existing browser tests now check every displayed reference row, sampled-choice language, and recomputation after workspace reload. The sixth hosted check compares the published before/after values to the independent fixture. Browser case counts remain 66 per browser; existing numerical and model-quality gates are unchanged.


### Selected-token head scan

Two new engine cases (89 total) compare all 48 trials in two independent PyTorch contexts (unedited and two inherited head edits), including probabilities, ranks, choices, tensor cleanup, deterministic sorting, source immutability, and cancellation before late publication. Two browser cases (68 per browser) verify all original-context rows, sorting, flow/draft selection, probability preview, branch choices, mobile containment, immediate cancellation, inherited edits, and navigation cleanup. The sixth hosted check scans a nested token, verifies the captured-head Normal result, selects it, and branches through the existing saved-workspace flow. No numerical tolerances or checkpoint-quality criteria change.

### Two-head story comparisons

Six additional engine cases (95 total) verify joint distributions, strongest alternatives, individual effects, interaction, and four independent six-token PyTorch rollout pairs. Coverage includes both duration policies, inherited/replaced heads, source expiry, seeded shared draws, atomic failure, cancellation, malformed/overlapping proposals, tensor cleanup, and populated version 1 workspace migration. Two new browser cases (70 per browser) exercise keyboard pair selection, removal/strength replacement, the two-head limit, individual/joint values, mobile containment, both durations, nested token inspection/further branching, frozen captured plans, reload, and workspace export/import. The sixth hosted check now measures and generates a two-head identity pair, replays its computation, and verifies its saved workspace record.

Local verification for this slice: all 95 engine tests, 11 Chromium story/scan/workspace cases, both new pair cases in WebKit, and the extended hosted continuation check against the production preview pass. The earlier scan cases also passed in Chromium/WebKit. Type checking, production build, and shipped/build artifact audits pass. Reference probabilities use the existing five-decimal numerical comparisons; no tolerances or checkpoint-quality criteria changed. CI and deployed release evidence are checked separately from these local results.

### Pair-scan story exploration

Six new engine cases (101 total) cover exhaustive head/strength enumeration, 720 independent PyTorch joint probability/rank/choice results in four complete scan contexts, canonical mixed-strength assignments, stable sorting, inherited-edit replacement, source replay/immutability, global seeded choices, cancellation, and retained-tensor cleanup. The scan runtime owns one exact-prefix baseline and releases every trial descriptor; failure, changed-prefix rejection, normal completion, and worker termination are covered separately. CPU replay checks compare CPU-recorded choices; independent PyTorch parity retains the existing five-decimal checks, without widening the replay tolerance.

Two new production browser cases (72 per browser) verify every default-scope row, sorting, strength-scope counts, the strongest pair's preview and independent six-token story paths, reload/replay, keyboard operation, narrow-screen containment, competing-scan/generation exclusion, usable partial mixed-strength results, and cancellation during deeper/source-token navigation. The sixth hosted check now completes a 120-pair Normal scan, selects a measured pair, then generates, inspects, and reopens its saved comparison. Raw scan rankings remain temporary. Existing numerical and checkpoint-quality boundaries are unchanged.

The first pair-scan release attempt passed 143/144 browser cases, including every new pair-scan case in Chromium and Firefox. An existing Firefox attention-routing check still showed a uniform redirect after attempting source activation. The setup now waits for the requested completed head, activates the source with the native button's keyboard action, and asserts its pressed state before checking the computed route. Five repeated runs per Chromium/WebKit pass; numerical and no-recomputation assertions remain intact.


### Head-pair interaction ranking

The pair scanner measures each proposed head alone and both together on the same source prefix, then ranks the joint-minus-sum interaction by absolute magnitude or either sign. All previous joint-change rankings remain available. Independent PyTorch fixtures now include both single-head probabilities, their summed changes, and interaction for all 720 joint rows across four unedited/inherited/mixed contexts, plus six-token continuations for each strongest-interaction pair. Every underlying probability keeps its five-decimal parity check. The interaction error budget is the sum of the four observed probability errors, plus floating-point arithmetic round-off; browser reference checks include displayed rounding. CPU-to-CPU source replay remains unchanged at 1e-6.

Engine coverage additionally checks exact additivity, either interaction sign despite the opposite joint-change sign, a strong interaction with a small net effect, all six ranking modes, canonical ties, reuse of exactly 48 unique individual trials across 1,080 all-strength pairs, and no partial row when either individual trial is cancelled or fails. The browser scan workflow checks each displayed metric, signed/absolute interaction sorting, strongest-interaction preview and independently verified story paths, reload/replay, inherited settings, partial results, navigation, and narrow-screen containment. The hosted continuation check verifies the default interaction ranking, signed sort selection, identity interaction, and saved pair branching. Local, CI, and deployed results are reported separately. Existing context, quality, and physical-device performance limits remain unchanged.

Local verification: 102 engine tests across 25 files, six Chromium head-scan/branch cases, both pair-scan cases in WebKit, and the extended hosted continuation case against the production preview pass. Type checking, production build, and public/build model-artifact audits pass. Full CI and deployed-release checks are verified separately.


### Head-pair interaction heatmap

Three additional engine tests (105 total across 26 files) verify the exact map coordinates and row/column strengths of all 720 independently measured joint/individual/interaction rows, mirrored same-strength pairs, both directed mixed assignments, excluded diagonal cells, and source-data immutability. All nine strength views together retain all 1,080 all-strength trials without aggregation. Partial maps distinguish an unmeasured reverse assignment from a measured zero and retain one symmetric scale across views.

The existing two pair-scan browser cases now verify all 240 displayed default-scope cells against the independent rows, keyboard navigation/selection, mirrored highlighting, narrow-screen scroll containment, unchanged ranked sorting, and choosing the strongest interaction from the heatmap through preview, six-token branching, download, reload, and computation replay. The inherited-source/partial case verifies exact strength transposition, ranked-list/heatmap selection agreement, all-strength filtering, an out-of-view selected pair, and cleanup on restart/navigation. The hosted continuation check selects its verified two-head identity pair from the published heatmap, then previews, generates, saves, and reopens the comparison. Browser case counts remain 72 per browser; no numerical or model-quality gates change. Local, CI, and hosted results are checked separately.

Local engine verification passes all 105 cases. An initial run alongside browser inference exceeded the existing exhaustive-reference timeout; the isolated complete suite passed with all time limits and numerical thresholds unchanged. The new mapping/contrast checks, production build, type checking, and shipped/build artifact audits also pass. Browser checks verify the heatmap workflow in Chromium and a phone-sized touch context, including 44-pixel targets and internal scrolling. Both WebKit pair-scan cases and the extended hosted continuation case against the local production preview pass; deployed release evidence is checked separately.


### Comparing story-token interaction maps

Five new engine cases verify exact source reconstruction from root/saved/nested-capable paths, same-scope head/strength joins, B-minus-A subtraction, identity and swapping, signed/absolute ranking, missing trials, source snapshot isolation, sequential prediction scopes, cancellation, and second-source replay failure. `scripts/export_story_heatmap_comparison.py` supplies two independent 120-pair PyTorch maps at global step seven after a muted-head fork diverges, with individual/joint probabilities, interaction, and all 120 differences. The existing numerical engine suite and new browser parity checks retain the per-probability five-decimal comparisons. Interaction differences propagate the eight measured probability errors; displayed values also allow rounding. CPU source replay remains at 1e-6.

Two production browser cases verify full A/B/difference maps, shared A/B scale, every matched difference, source/prefix/edit provenance, pair selection across maps, ranking, swapping without inference, downloadable records, narrow-screen containment, unchanged saved stories, and navigation. Delayed-worker coverage checks unmatched partial maps, source-change cancellation, close/reset, strength views, and mutual exclusion with existing scans/generation. The hosted continuation check scans an identity pair of saved paths, selects its zero difference, and retains the saved branching workflow. Full counts become 110 engine tests across 27 files and 74 browser cases per browser. Local, CI, and published-site evidence are checked separately. Model-quality, context, educational, and physical-device performance boundaries are unchanged.

Local verification passes all 110 engine tests, four focused Chromium cases (both comparison cases and both existing pair-scan cases), both comparison cases in WebKit, the full comparison in phone-sized Chromium touch emulation, and the extended hosted continuation check against the production preview. The final delayed-worker Chromium check also verifies that comparison scanning pauses an active branch. Type checking, production build, artifact audits, and whitespace checks pass. Touch emulation is not physical-device performance evidence; full CI and deployed checks remain separate release evidence.


### Shared pair branching from interaction differences

Four new engine cases compare all four continuations against independent shipped-weight PyTorch outputs for sustained edits, first-prediction edits, different inherited/expired settings and source steps, and an identity pair. Synthetic cases verify each source's own seed/global counter, identical proposals/durations/budgets, snapshot isolation, atomic failure/cancellation across both sources, exact paused resume, independent EOS, and malformed pause alignment. A fifth engine case validates version 3 shared links, migration from version 2, exact parent-derived source selections, unique membership, matching proposals/policies, aligned frames, and refusal of invalid imports. Totals become 115 engine cases across 28 files. `scripts/export_story_shared_forks.py` exports the four-path fixtures; existing five-decimal independent probability checks and 1e-6 CPU replay checks remain unchanged.

The full heatmap browser workflow now selects the strongest measured difference pair, generates both sources with both duration policies, compares every output token/probability with the independent reference, checks source immutability, exports both linked results, reloads/reopens them, inspects a generated computation, branches further, and round-trips a workspace including invalid-link rejection. A new delayed-worker case covers second-source worker failure with no half-step publication, stop/close/reset during an incomplete step, persisted pause/resume after reload/import, prevention of independent member advancement, and keeping an open member view current during shared generation. The three comparison cases bring the suite to 75 cases per browser. The hosted continuation check also generates an identity pair for both saved sources, checks all four paths, reloads their shared result, and inspects a generated token. Local, CI and deployed evidence are reported separately; model quality, context, and physical-device claims are unchanged.

Local checks pass all 115 engine cases with four test workers, all three Chromium comparison cases, the complete four-path flow in phone-sized Chromium touch emulation (including 44-pixel controls), and the extended hosted continuation case against the production preview. An initial unrestricted-parallel engine run exceeded the existing five-second residual-sweep deadline; the bounded complete run passed with every time limit and numerical threshold unchanged. Type checking, production build, artifact audits, and whitespace checks pass. WebKit, full release CI, and the published-site checks are verified separately. Emulation does not establish physical-device performance.

The first shared-branch release attempt passed all 115 engine cases and 149 browser cases, including the full four-path reference workflow in both Chromium and Firefox. The overall 20-minute verification-job deadline interrupted the final Firefox lifecycle case less than two minutes after it started; no completed browser case failed. The verification-job budget is now 25 minutes to accommodate the larger 150-case suite. Individual browser/engine deadlines, probability tolerances, coverage, and release gates are unchanged. The full rerun and deployed checks remain required.

Subsequent fresh-runner retries hit slow Azure Ubuntu archive downloads before browser tests started (for example, a 3.5 MB font package took roughly 55 seconds, with installation still downloading after six minutes). CI now prefers the standard HTTPS Ubuntu archive, then the security archive, retaining Azure as the final fallback in the runner's existing apt mirror list. The same setup applies to verification and hosted checks. Mirror URLs and apt mirror-list priority syntax follow the [official runner image configuration](https://raw.githubusercontent.com/actions/runner-images/main/images/ubuntu/scripts/build/configure-apt-sources.sh); package signature verification and all test/release gates remain enabled. These setup retries do not count as browser-test passes.


### Interaction trajectories along shared continuations

Four new engine cases bring the suite to 119 cases across 29 files. `scripts/export_story_interaction_trajectories.py` independently measures 88 token points along both reference/edited paths of four shared scenarios: sustained edits, first-prediction edits, inherited/expired settings with different source steps, and an identity pair. Coverage checks each source/individual/joint probability, signed interaction, expiry, actual-path replay, exact prefixes, snapshot isolation, cancellation/failure without partial point publication, and retained-tensor cleanup. Display-precision trend checks cover growth, fade, reversal, zeroes, and missing offsets. The per-probability five-decimal parity checks and 1e-6 CPU replay threshold remain unchanged; interaction comparisons propagate the four probability errors.

Two additional browser cases bring the suite to 77 cases per browser. They check complete reference/edited token tracks against PyTorch, shared scales, actual reversals, synchronized token details and keyboard selection, expiry, narrow-screen scrolling/touch targets, exact captured exports, unchanged saved stories, reload and computation replay. Delayed-worker checks cover partial results, generation/scan exclusion, resume invalidation, close/reset cleanup, and reimport without stale points. Browser inputs are generated with `npx vitest run --config vitest.trajectory-fixture.config.ts` from real CPU inference and validated workspace ancestry; independent numerical expectations still come from the separate PyTorch exporter. The sixth hosted case measures the saved identity pair along both continuations, checks its probabilities and zero interaction, selects a token, and retains the existing flow inspection checks. Workspace version 3 and model-quality/context/physical-device limits remain unchanged.

Local bounded-worker verification passes all 119 engine cases, type checking, production build, and public/build artifact audits. Both new Chromium trajectory cases and all three existing Chromium heatmap/shared-branch cases pass. Browser screenshots were reviewed at desktop and 390-pixel widths. WebKit, touch emulation, full release CI, and deployed-release evidence are checked separately.
