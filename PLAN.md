# Transformer Microscope implementation plan

Status: Milestones C/D and the core Milestone E experiment workspace are implemented and committed. Milestone F has local named-host latency measurements, artifact audits, and a passing remote CI workflow for Chromium CPU/WebGL and Firefox CPU. The HTTPS Pages release and installed Safari CPU/WebGL smoke checks pass. An opt-in on-device validation harness, a physical-device protocol, an educational evaluation protocol, and a session page that runs it are implemented. The physical-phone run is deferred by project decision (not passed); educational sessions with participants are next. See `docs/validation.md`, `docs/retrieval.md`, `docs/benchmarks.md`, and `docs/deployment.md`.

This plan incorporates the product concept, intervention-engine specification, and design review. It prioritizes a verified head-scaling experiment before expanding into the full Explore, Experiment, and Train workspaces. Performance targets and numerical tolerances below are requirements to validate, not measured results.

## 1. First release and success criteria

The first release opens a prepared example, runs a real transformer locally, lets the user select an attention head and change its strength, and compares the resulting prediction with an immutable baseline. Users can inspect the tensors that explain the intervention and reset to the baseline.

The release must pass two independent gates:

1. **Numerical correctness:** Python reference, TypeScript full execution, and cached suffix execution agree within documented tolerances on fixed fixtures.
2. **Educational usefulness:** a trained checkpoint and prepared examples demonstrate measured, understandable dependencies, including both an intervention with an effect and a control with little or no effect.

The first release includes:

- One pretrained, controlled-task checkpoint with a small, explicit tokenizer.
- A prepared example plus editable input within that model's supported domain.
- Token and prediction-position selection.
- Layer/head selection, attention inspection, and head-output scaling from 0 to 2.
- Scaling at all positions or explicitly selected positions.
- Baseline and modified prediction comparison with reset.
- Selective inspection of baseline and modified activations.
- CPU fallback, one validated accelerated backend, and static deployment.

Defer the full training UI, free-form architecture configuration, text-model training at larger scale, attention editing, arbitrary activation replacement, cross-run patching, ablation matrices, persistent experiment libraries, offline service workers, and generation. The computation engine must still pass a small gradient and optimizer test early so later training remains feasible.

## 2. Implementation principles

- Model execution supplies every displayed numerical value.
- An intervention changes activations during execution; it never mutates model weights or the baseline.
- A correct full forward pass is always available. Caching changes performance, not semantics.
- Python provides independently implemented reference fixtures. Do not generate expected outputs from the TypeScript implementation under test.
- React sees serializable records and copied numerical slices, never tensor handles.
- One worker owns model weights, backend initialization, execution, and tensor lifetimes.
- Keep the public engine interface small. Execution planning, caching, and tensor ownership are internal implementation details.
- Start with a fixed execution order and a small internal hook registry. Do not build a generic graph scheduler or a backend abstraction over TensorFlow.js.
- Do not silently substitute a different model to meet a performance target. Any model or context change must be visible and must establish a new baseline.

## 3. Freeze the numerical contract

Record these choices in `docs/model-contract.md` and encode the machine-readable fields in the model manifest before writing the forward pass.

### Initial architecture

- Decoder-only transformer with pre-layer normalization and final layer normalization.
- Learned token embeddings plus learned absolute position embeddings.
- Separate Q, K, and V projections, each with bias.
- Causal attention including the current token; scale scores by `1 / sqrt(headDimension)`.
- Head output layout `[batch, position, head, headDimension]` after `A × V`.
- Concatenate heads in ascending head order before output projection; apply output-projection bias once.
- Two-layer MLP with biases and tanh-approximate GELU, with its exact formula recorded in the contract.
- LayerNorm over the final feature axis, population variance, affine scale/bias, epsilon `1e-5`.
- Tied token embedding and unembedding weights; no unembedding bias initially.
- No dropout in the initial architecture, including the small training probe.
- Float32 serialized weights. Record actual backend precision separately from serialized dtype.
- Internal indices are zero-based; the UI may display one-based layer/head labels.

### Inputs and outputs

- Batch size 1 for interactive inference; full-sequence execution without a KV cache.
- The logit vector at position `i` predicts the next token after that position.
- A comparison specifies `predictionPosition`, defaulting to the last input position, and optionally a target vocabulary token.
- Initial interactive tokenizer: explicit symbols with deterministic whitespace separation and a dedicated beginning-of-sequence token. Do not silently lowercase or normalize user input.
- Unsupported symbols and overlong inputs produce actionable validation errors. Empty text maps to the documented beginning-of-sequence input.
- Compare distributions at temperature 1 before any sampling filter.
- Use natural logarithms for log probabilities and Jensen–Shannon divergence; label their units as nats.
- Rank ties use ascending token ID for deterministic presentation.

### Model artifact

Use `manifest.json`, `tokenizer.json`, and `weights.bin`. The manifest records:

- Schema and architecture versions, architecture dimensions, and all numerical conventions.
- Tensor names, shapes, byte offsets, byte lengths, float32 little-endian encoding, and matrix orientation.
- A content hash for the weights and tokenizer; model revision identifies the exact artifact contents.
- Training configuration, seed, data-generator version, evaluation summary, and provenance.

Store dense linear weights in `[inputDimension, outputDimension]` order and explicitly transpose PyTorch linear weights on export. Represent tied weights once. Validate sizes, shapes, hashes, finite values, and configuration compatibility before accepting a model.

## 4. Run and intervention contracts

Define shared TypeScript types and runtime validators before connecting the worker or UI.

### Public engine interface

The engine client exposes `loadModel`, `runBaseline`, `compare`, `inspectActivation`, `releaseRun`, and `dispose`. Tests and the UI use the same behavioral interface; numerical kernel tests may use internal seams where necessary.

- `runBaseline` returns an immutable run ID, execution identity, input token IDs, and prediction summary.
- `compare` takes a baseline run ID, interventions, prediction position, optional target token, and display limit. It returns a modified run ID, comparison metrics, result provenance, and execution metadata.
- `inspectActivation` accepts either baseline or modified run IDs and a typed activation reference. It returns the realized shape, axes, and copied values for a bounded selection.
- `releaseRun` is idempotent. Releasing a baseline cannot silently break an existing modified run; modified runs retain their own immutable input/intervention descriptors and can replay after the source baseline is released.
- `dispose` releases the runtime's owned resources and rejects pending requests with a typed error.

Keep immutable run descriptors separate from cached tensor materializations. A descriptor records the exact model revision, tokens, intervention list, and execution settings needed for recomputation. Cache eviction may remove tensors while a run remains inspectable through selective recomputation. Explicit release ends that guarantee. Model reload or worker restart invalidates the current session's handles with a clear error.

### Typed activation references

Support a small stable set of named stages: embedding output; block input; normalized attention input; Q/K/V; pre-mask scores; masked scores; attention weights; raw head output; modified head output; projected attention; attention residual; MLP output; block output; final normalized state; logits.

Distinguish raw and modified head outputs explicitly. An attention-cell inspection should retrieve the selected Q/K vectors, dot product, scaling, mask status, and softmax result. Recompute uncached values with the requested capture selection rather than retaining all activations indefinitely.

### Head scaling

The only initial intervention is `head.scale`, with layer, head, token selection, and finite factor in `[0, 2]`.

- Apply scaling after `A × V` and before output projection.
- Validate integer indices, sequence bounds, duplicate token positions, and model identity before allocating tensors.
- Reject overlapping selections on the same layer and head.
- Treat an empty intervention list or empty token selection as an identity operation after validation.
- Remove factor-1 operations when planning effective work, while preserving the submitted experiment description.
- For multiple layers, apply hooks in forward execution order. Never reuse downstream baseline tensors after an earlier edit.
- Implement a single-head UI initially; validate multiple non-overlapping interventions through engine tests.

### Execution identity and lifecycle

Identity includes model revision, architecture version, exact token IDs, position/mask policy, evaluation mode, backend, effective precision information, and runtime generation. A backend name alone is insufficient to identify tensor ownership after a restart.

Use structured error codes for invalid input, incompatible model, released run, stale session, unsupported backend, resource exhaustion, and numerical failure. Active executions pin their resources until completion; eviction and release cannot dispose tensors still in use.

## 5. Project structure and tooling

Use React, TypeScript, Vite, and TensorFlow.js Core with explicit backend dependencies. Pin dependency versions when creating the project. Use Python/PyTorch for training, export, and reference fixtures. Keep the first UI's state local to its feature; introduce shared state tooling only when coordination warrants it.

```text
docs/
  model-contract.md
  validation.md
  benchmarks.md
src/
  engine/
    contracts.ts
    model.ts
    weights.ts
    interventions.ts
    execution.ts
    runs.ts
    inspection.ts
    comparison.ts
  workers/
    model.worker.ts
    protocol.ts
    client.ts
  features/experiment/
  App.tsx
scripts/
  reference_model.py
  export_fixtures.py
  train_demo.py
  export_model.py
tests/
  fixtures/
  engine/
  browser/
public/models/demo/
public/examples/
```

Use Vitest for pure logic and CPU numerical tests, and Playwright for the worker, real browser backend, UI, and production bundle. Run Python reference generation explicitly; routine tests consume checked-in, versioned fixtures. Regenerating fixtures requires reviewing the underlying numerical change.

## 6. Milestone A — Reference model and browser feasibility

### Work

1. Scaffold the TypeScript application and Python environment, pin dependencies, and add basic test commands.
2. Write the model contract and artifact schema.
3. Implement the independent PyTorch reference with selected activation capture and head scaling.
4. Export a tiny correctness model: 2 layers, 2 heads, model dimension 8, MLP dimension 16, vocabulary 16, context limit 8.
5. Export actual weights, token IDs, intermediate tensors, logits, loss, selected gradients, and one optimizer update. Do not rely on cross-language seed equivalence.
6. Build a minimal production-bundled worker probe for the intended operations: lookup, normalization, matrix multiplication, masking, softmax, GELU, backward pass, optimizer update, and asynchronous readback.
7. Exercise CPU and the intended accelerated backend. Test worker availability on desktop Chrome, Firefox, and Safari, and record unsupported combinations.

### Acceptance

- The deterministic two-head arithmetic example reproduces baseline `[10.5, 9.5]` and half-scaled output `[9.5, 6.5]`.
- Fixture metadata records generator versions, dimensions, tensor layouts, and weight hashes.
- The production worker can execute the operation probe and return finite results on the CPU fallback.
- At least one intended accelerated configuration completes the probe; failures are documented before UI assumptions are made.
- Numerical comparison and performance measurements include browser/device/backend information.

This milestone is complete without a trained product checkpoint or polished visualization.

## 7. Milestone B — Correct full execution

### Work

1. Implement artifact validation and loading.
2. Implement the transformer using one shared set of mathematical operations for inference and differentiable training.
3. Add head scaling at the declared hook and selective activation capture.
4. Implement run descriptors, typed inspection, full-vocabulary comparison metrics, and explicit disposal.
5. Implement causal reachability reporting for the initial architecture.

### Acceptance

- Every selected stage matches the independent PyTorch fixture within documented absolute/relative tolerances.
- Cover sequence lengths 1, 2, 5, and 8, both layers and heads, factors 0, 0.5, 1, and 2, and single/all/disjoint token selections.
- Test factor 1 through forced full execution, not only the identity shortcut.
- Earlier-position logits remain unchanged by later-position edits.
- Editing an earlier position's final-layer head output leaves the last-position prediction unchanged: no later attention layer can carry the edit across positions.
- Head scaling preserves same-layer attention weights and unselected raw head outputs.
- Output-projection bias is applied exactly once and is not scaled with a head.
- Selected gradients agree with PyTorch and finite differences on a tiny, well-conditioned fixture. A matching optimizer step is verified with explicit optimizer hyperparameters.
- Input and intervention validation failures do not retain newly allocated tensors.

For reachability, distinguish an absent computational path from a reachable output whose measured change is small. A multi-intervention experiment must not be labeled unreachable if any effective intervention has a path to the selected prediction.

## 8. Milestone C — Cache and suffix execution

Implemented and validated: 319 direct cached/reference cases, 29 engine tests, and 10 production browser tests. CPU and unpacked WebGL in Chromium and WebKit each pass cached/full equivalence and 1,000 sequential comparisons with bounded retained tensors. Detailed coverage and measurements are in `docs/validation.md` and `docs/benchmarks.md`.

### Work

1. Retain raw head outputs and attention residuals at eligible layers. Add other retained tensors only for a demonstrated use.
2. Implement a pure planner with `identity`, `full`, and `cached-suffix` strategies.
3. For one affected layer, compute `Y' = Y + sum((factor - 1) × Z_head × W_output_head)`, including token masks, then execute the MLP and remaining blocks once.
4. For cross-layer edits, initially use the full forward path.
5. Add a logical tensor budget, active-run pinning, eviction, and selective inspection recomputation.
6. Make ownership consistent: internal helper results must have a documented ownership rule without making callers guess whether the result aliases an input.

### Acceptance

- Cached and forced full execution agree across the Milestone B cases and multiple non-overlapping edits in one layer.
- Compare intermediate modified residuals as well as final logits.
- Repeating factor 0.5 independently produces the same result; resetting to 1 restores the baseline.
- Changed weights, tokens, runtime generation, or backend cannot reuse invalid tensor caches.
- Active tensors survive concurrent inspection/release/eviction requests until their consumers finish.
- After warm-up, 1,000 sequential comparisons show bounded retained tensor counts; releasing all runs returns to the expected loaded-model baseline.
- Resource tests include failed requests, evictions, and model reloads, not just successful comparisons.
- Inspection of a modified run returns modified values, including after materialization eviction.

Keep `tf.tidy()` callbacks synchronous. Read values asynchronously outside the scope and dispose retained outputs in `finally` blocks. Report logical payload sizes separately from backend memory estimates.

## 9. Milestone D — A checkpoint worth exploring

This can begin after the architecture contract is stable; it does not require the completed UI.

### Work

1. Create a controlled associative-retrieval dataset: earlier symbol/value bindings followed by a query for one binding.
2. Randomize assignments so the final local token context alone cannot determine the answer. Define unambiguous binding and query rules.
3. Separate training and held-out generated sequences, recording generator seeds and split rules. Include length or combination generalization separately from ordinary held-out accuracy.
4. Train an initial candidate with 2 layers, 2 heads, model dimension 32, MLP dimension 128, and context up to 32. Adjust capacity only in response to recorded evaluation results. The initial candidate plateaued near 61%; the accepted model uses 2 layers, 4 heads, dimension 64, MLP 256, and a lower learning rate (see `docs/retrieval.md`).
5. Compare held-out target accuracy/loss against explicit chance and local-context baselines.
6. Measure head interventions and select at least three prepared experiments: a measurable effect, a reachable small-effect control, and a structurally unreachable control.
7. Export the checkpoint, tokenizer, evaluation results, and experiment recipes.

### Acceptance

- The model demonstrates use of the intended earlier context on held-out examples.
- Each recipe records input, prediction position, target token, intervention, checkpoint revision, baseline metrics, modified metrics, and a bounded interpretation.
- Effect sizes exceed the measured numerical discrepancy for that fixture/backend where an effect is claimed.
- Controls are retained as useful examples rather than filtered out because they look less dramatic.
- No head receives a semantic label solely from one prompt or ablation.
- The browser reproduces the selected examples within its documented tolerance.

If the checkpoint fails this gate, revisit data, objective, or capacity before broadening the UI. Correct arithmetic alone does not satisfy this milestone.

Milestone D is implemented: the accepted checkpoint answers 453/453 held-out three-binding sequences correctly, with separately reported length generalization. Three validation-selected recipes reproduce on browser CPU/WebGL. Training reproduced identical weight, tokenizer, and manifest hashes. `docs/retrieval.md` and `docs/validation.md` record limitations and evidence.

## 10. Milestone E — Interactive experiment screen

### Work

1. Implement typed worker requests and responses, runtime-generation IDs, structured errors, and explicit lifecycle commands.
2. Serialize model mutations with execution. Keep one active comparison and only the latest pending comparison; settle superseded client requests predictably.
3. Build model loading/error states, prepared-example selection, validated input, token selection, and prediction-position selection.
4. Add layer/head controls, token scope, slider, numeric factor entry, and baseline reset.
5. Show baseline/modified predictions using a stable union of relevant tokens, including the target. Compute all metrics from the complete vocabulary.
6. Add the attention view, attention-cell arithmetic, and raw/modified head-output inspection.
7. Label every completed result with the intervention that produced it. Mark older results while a new value is pending.
8. Use a throttled preview and latest-pending scheduling; request the final value on interaction completion. Measure the cadence before fixing its interval.

### Acceptance

- A user can complete the prepared experiment without configuring a model or reading implementation documentation.
- Slider movement remains responsive while computation runs, and stale responses never replace the latest requested result.
- Reset works while a prior request is in flight.
- The UI explains why head-output scaling leaves the same-layer attention map unchanged.
- Displays distinguish unreachable predictions, small measured effects, pending results, and errors.
- Controls work with a keyboard; heatmap cells have an accessible numerical inspection alternative and do not rely only on color.
- A focused stacked layout supports mobile inspection without silently changing the experiment.

## 11. Milestone F — Release verification and deployment

### Work

1. Build production assets using the actual repository deployment subpath; use base-aware model URLs and bundled module workers.
2. Run browser tests against the production build, including model loading, inspection, intervention, reset, error recovery, and worker restart.
3. Validate desktop Chrome, Firefox, Safari, and at least one physical lower-powered mobile device. Record versions and backend selection; browser emulation is not evidence of physical-device GPU performance.
4. Benchmark cold load, warm baseline inference, full versus cached comparison, inspection readback, and end-to-end input-to-result latency.
5. Add GitHub Actions for tests and static deployment once the repository and hosting are configured.

### Acceptance

- All model computation runs locally after static assets are loaded; no inference service is required.
- Model and worker assets load correctly from the production subpath.
- The CPU fallback provides a usable reduced-performance experience with visible backend status.
- Report warm comparison median and p95 latency by device, backend, model, sequence length, intervention layer, and capture mode.
- Treat sub-200 ms warm end-to-end comparison latency on a named desktop configuration as an initial target, including scheduling and readback. Investigate misses without presenting the target as an established capability.
- Benchmark early, middle, and late layers. Expect larger opportunities for suffix reuse at later layers, but require measured evidence before claiming a speedup.
- Document numerical limitations, unsupported browser/backend combinations, and resource limits.

## 12. Validation policy

Maintain `docs/validation.md` with exact fixture definitions, coverage, provenance, and results.

- Separate exact invariants from floating-point parity and empirical performance expectations.
- Use `abs(actual - expected) <= atol + rtol * abs(expected)` for numerical comparisons, with explicit checks for finite values where expected. Masked score sentinels are validated according to their special semantics.
- Record maximum absolute error and a scale-aware error summary by stage. Establish tolerances using independent reference fixtures and actual backend behavior; do not widen them automatically after a failure.
- Test ill-conditioned inputs, near-zero values, large-but-valid activations, and negative control fixtures in addition to normal random inputs.
- Use finite differences as a gradient diagnostic with a documented perturbation size, not as an arbitrary universal threshold.
- Test ranking stability separately from numerical parity when logits are nearly tied.
- Calculate memory baselines after warm-up; backend allocation estimates may be incomplete or affected by pooling.
- Preserve benchmark configuration and raw results so later regressions are comparable.

## 13. Expansion after the first release

Proceed only after the first release passes both the correctness and educational gates.

1. **Broader exploration:** embeddings, residual views, richer computation traces, and a validated text checkpoint with a fully specified subword tokenizer. *Started 2026-10-12 by project decision before the educational gate:* a residual-stream read-out (logit lens) at the prediction position and an input-embedding view, both verified against independent PyTorch values (`docs/computation-inspection.md`). The validated subword text checkpoint remains. Its proposal (`docs/text-checkpoint-proposal.md`) is accepted, and the data, tokenizer, training, and probe pipeline is implemented with Python/TypeScript tokenizer parity on every validation story. Both permitted training runs failed the name-copying probe (24,000 steps: 77.5%; the one retry at 48,000 steps: 56.0%; threshold 80%), so the original text-checkpoint acceptance gate remains failed. At the project owner’s subsequent request, the 48,000-step checkpoint is shipped as an experimental real-text model, with BPE-aware loading, input, replay, and independent numerical verification (`docs/tinystories-deployment.md`). A revised probe set that evaluates the existing checkpoints without further training is drafted for approval (`docs/text-checkpoint-probes-v2.md`).
2. **Additional interventions:** residual replacement, normalized causal attention overrides, and compatible-run activation patching, each with its own semantics and full-versus-optimized tests. *Residual replacement (engine) implemented 2026-10-12:* `residual.replace` at `block.input` or `attention.residual`, full execution only, verified against independent PyTorch cases and a 64-case retrieval patching sweep (`docs/model-contract.md`, `docs/validation.md`). An on-screen patching mode copies a same-length source input's residual into the current input and is verified against the PyTorch sweep. *Attention overrides implemented 2026-10-12:* `attention.override` replaces one head's post-softmax row with a normalized causal distribution and is verified against PyTorch cases and a 64-case routing sweep; the screen offers a redirect mode. *Saving implemented 2026-10-12:* experiment files are now version 2, saving head scaling, attention redirects, and residual patches with their source input; version 1 files still import (`docs/experiment-format.md`). Optimized (cached) paths for replacements and overrides remain.
3. **Ablation explorer:** measured layer/head effects for a selected prediction, with explicit caveats about prompt dependence and non-additive interactions. *Implemented 2026-10-13:* a per-prediction scan of every single-head ablation with pairwise joint-versus-sum comparison, verified against PyTorch (`docs/computation-inspection.md`); dataset-wide context is in the head-coverage analysis.
4. **Training engine and workspace:** datasets, bounded training steps, loss/gradient inspection, pause/resume, and checkpoint comparison.
5. **Persistence and replay:** versioned experiment recipes and IndexedDB storage with explicit migration and quota handling.

For training, distinguish lightweight inspection snapshots from resumable checkpoints. A resumable checkpoint includes weights, optimizer moments, optimizer step, hyperparameters, data-sampler position, and random-generator state. Interventions run against a paused or immutable checkpoint; they cannot reuse caches while weights are changing.

## 14. Immediate next work

The default Visual flow now presents input, parallel attention heads, processing, and computed predictions with plain-language explanations. Head strength and baseline reset are available inside the diagram. Detailed preserves full inspection, interventions, tracing, replay, and runtime controls. See `docs/visual-flow.md`.

Complete the remaining Milestone F release gates. The trained checkpoint and three prepared recipes are now in the screen. Version 1 experiment export/import is implemented with checkpoint validation, complete logit snapshots, automatic installed-model selection, and live replay comparison. Attention-cell arithmetic, lazy projection/residual/MLP vectors, exact strength entry, and CPU worker restart are now implemented. Local Milestone F work is implemented: named-host timing records, artifact audits, and CI/deployment preparation. Remote CI now passes, including Firefox CPU on Linux; its worker WebGL context is unavailable and recorded explicitly. The verified build now deploys automatically to HTTPS Pages and passes exact asset/workflow checks. Installed Safari CPU/WebGL reference probes, baseline reset, inspection, CPU restart, and hosted CPU verification pass. The release-validation tooling is now in place. `validation.html` is an opt-in harness that records browser/OS, deployed commit, model revisions, actual backends, PyTorch probe and recipe parity, replay, resource cleanup, cold readiness, and warm interaction timings (DOM commit, frame opportunity, RPC round trip, and worker computation reported separately), and exports raw JSON. Failed and unsupported checks are explicit. It passes in CI Chromium/Firefox and against the deployed site; those are automation results. `docs/physical-device-protocol.md` adds the manual touch checklist and evidence fields, and `docs/educational-evaluation.md` defines participant sessions and fixed gate criteria with a findings template. No physical mobile device or participant session was available, so neither gate is met. A lost WebGL context previously left the screen on "Computing…" indefinitely. The worker now detects the loss, discards post-loss results, and ends the session; the screen offers a CPU restart, and the harness checks this path on each device.

On 2026-10-12 the project owner chose to skip the lower-powered phone run and proceed to educational evaluation. The physical-mobile gate is deferred, not passed; its harness and protocol remain ready. The opt-in session page `study.html` now runs the educational protocol. It locks each prediction before showing the live recipe result, records the displayed result, and exports per-participant JSON. `scripts/summarize_study.mjs` turns the records and human probe codes into the findings tables and gate tally.

Single-head ablation coverage across all 453 held-out inputs is now recorded (`npm run analysis:heads`, `docs/retrieval.md`). Ablating Layer 1 Head 4 changes the answer on 218 inputs and ablating Layer 2 Head 2 on 209. The large-effect recipe is near the extreme of that range, layer-1 position-0 ablations never change an answer, and final-layer position-0 ablations are exactly zero. The facilitator uses this spread in the C4 debrief, without presenting a head role.

Next actionable step: run five facilitated sessions with `study.html` (two on phones if possible). Code probes C1–C5 with two coders, summarize with `scripts/summarize_study.mjs`, and record the outcome in a findings document. Address any misconception seen twice and retest it with new participants. Wider trained-head/prompt coverage remains follow-up work; larger models, the training UI, new interventions, and the ablation explorer stay deferred until both gates pass.

No latency, checkpoint quality, browser compatibility, or numerical parity claim is complete until its associated evidence is recorded.

## Controlled continuation slice

At the project owner's request, generation expands beyond the original first-release scope. Implement bounded token-by-token TinyStories continuation with step/play/stop/reset, deterministic greedy or seeded temperature/top-p sampling, a fixed-context stopping policy, and optional original/edited head-output branches. Keep generated IDs exact, snapshot settings and interventions, isolate cancellation from the diagram, and verify rollouts against independent shipped-weight PyTorch fixtures. The contract and follow-up boundaries are recorded in `docs/controlled-continuation.md`; the earlier quality and educational gates are unchanged.


## Generated-token inspection slice

Completed continuation tokens open a read-only historical flow. Reconstruct the exact prefix before the selected token, replay its frozen greedy/seeded choice on the matching CPU checkpoint, and inspect the same stages/vectors as the main diagram. Edited-branch comparisons use the same edited prefix on both sides and state that distinction explicitly. Preserve the main prompt/experiment and recorded generation; dispose historical work on navigation, closure, or session changes. See `docs/controlled-continuation.md` for lifecycle, evidence, and remaining boundaries.
