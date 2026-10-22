# Transformer Microscope

A browser laboratory for inspecting transformer computations and measuring internal interventions. The first implementation includes a verified two-layer transformer, a dedicated computation worker, and a head-scaling screen.

**The default checkpoint learns a controlled symbol retrieval task.** It answers 453/453 held-out three-binding inputs correctly. It is not a general language model; accuracy drops with different binding counts. The untrained numerical verification checkpoint remains selectable.

The default model now accepts real English text with the TinyStories checkpoint. The retrieval and verification symbol models remain selectable.

Use **Watch a story grow token by token** to step, play, stop, or reset a continuation of the completed prompt. Choose the most likely token or repeatable seeded sampling, and optionally compare the original story with a captured head-output edit. Generation stops at EOS, the chosen token budget, or the checkpoint's 64-token context; it does not drop earlier text. Select any numbered generated token to inspect the exact preceding context, captured edit, probability, and stage values in a read-only flow. Choose a head in that flow and use **Compare from this token** to continue two stories from the same prefix, applying your edit once or throughout the continuation. Continuation and branch downloads preserve exact token IDs and settings. Details: `docs/controlled-continuation.md`.

The default **Visual flow** follows symbols through embeddings, parallel attention heads, per-position processing, and prediction. Click a stage to select its existing controls, or switch to **Detailed** for the full computation graph, traces, and numerical inspectors. Both views share the same experiment. See `docs/visual-flow.md` for diagram semantics and accessibility.

The computation workspace presents editable bindings, a clickable transformer flowchart, contextual head-strength controls, and a persistent baseline/modified prediction card. Selecting a node inspects the completed experiment; an intervention is applied when its controls change. Use **Edit tokens** for raw symbol inputs. See `docs/frontend-workspace.md` for the interaction and verification details.

**Try an example** starts one of three prepared experiments: a large answer-changing edit, a tiny reachable effect, or a local edit with no route to the prediction. The completed-example guide shows the actual measured result and provides **Inspect the edited head** and **Trace this example** actions. Input and source drafts remain separate from the completed guide.

**Follow the change** traces a completed head-scaling, attention-redirect, or residual-copy intervention through the graph, with separate route and measured-difference badges, guided steps, cross-position attention handoffs, and diagnostic residual readouts. Redirect attention by selecting query and source tokens; future sources are unavailable. The graph separates the edited weights from the resulting head output. To copy a residual, select a junction, compare aligned source/current tokens, and press **Copy from source**. Source and position choices remain drafts until applied. Navigation reuses one batched measurement and keeps the prediction visible.

## Run

Use Node.js 24 and npm:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/transformer-scope/`. The trained input format is `A = 0 ; B = 3 ; C = 1 ; ? A =` with spaces around every symbol. Keys A–D are distinct; values 0–3 may repeat. Choose a prepared experiment or write your own input. The selected verification checkpoint accepts A–O with at most seven symbols. Choose a prediction position, layer, head, token scope, and head strength. The baseline remains immutable; factor 1 restores it. Export a completed experiment (head scaling, attention redirect, or residual patch) as JSON, then import it to restore settings and compare recomputed predictions with saved values. See `docs/experiment-format.md` for the versioned format (version 2; version 1 files still import).

Both shipped checkpoints default to CPU based on the recorded cold/warm browser measurements. WebGL remains selectable; other larger checkpoints attempt WebGL with CPU fallback. `?backend=cpu` or `?backend=webgl` chooses the initial backend explicitly. The verification button checks actual browser forward values, gradients, and an Adam update against exported PyTorch results.

## Check

```sh
npm test
npm run test:data # requires the pinned Python environment
npm run build
npx playwright install chromium firefox webkit
npm run test:browser
npm run benchmark:cache
```

To run a particular installed browser, use `npm run test:browser -- --project=chromium` (or `webkit`/`firefox`). The local Firefox runtime currently fails to launch its profile on this host; that is recorded as an environment limitation, not a passing browser check.

Browser tests use the production build served at `/transformer-scope/`. Engine tests consume checked-in independent reference fixtures; they do not require Python.

## Validate on a device

Open `validation.html` under the deployment (for example https://jtmcbride.github.io/transformer-scope/validation.html), describe the device, and press **Start validation**. It records the deployed commit, model revisions, actual backends, PyTorch reference checks, the prepared recipes, replay, cold readiness, and warm interaction timings, then exports raw JSON. It runs only when started and is not linked from the application. Follow `docs/physical-device-protocol.md` for the full physical-device procedure.

## Run an educational session

Open `study.html` under the deployment (https://jtmcbride.github.io/transformer-scope/study.html) and follow `docs/educational-evaluation.md`. The page locks each prediction before showing the live result, then exports one JSON record per participant. Summarize coded sessions with `node scripts/summarize_study.mjs --codes codes.json sessions/*.json`.

## Regenerate reference fixtures

Use Python 3.12 and uv:

```sh
uv sync --locked
npm run fixtures
npm run fixtures:exploration   # residual read-out and embedding values for the trained checkpoint
npm run fixtures:residual      # residual-replacement cases and the retrieval patching sweep
npm run fixtures:attention     # attention-override cases and the retrieval routing sweep
npm run fixtures:ablation      # single-head and pairwise ablation logits
npm run fixtures:trace         # head-scaling stage summaries and diagnostic readouts
npm run fixtures:attention-trace # redirect rows, stage summaries, and diagnostic readouts
npm run fixtures:residual-trace # copied residuals, stage summaries, and diagnostic readouts
```

Fixture regeneration is deliberate and must be reviewed when the numerical contract changes. The exporter records its seeds and PyTorch version, produces 84 original forward/intervention cases and 261 additional three-layer cases at seeds 1, 7, and 42, and saves loss, all gradients, and one Adam update. Weights and expected values come from the independent implementation in `scripts/reference_model.py`.

## Train the retrieval checkpoint

```sh
uv sync --locked
npm run test:data
npm run train:retrieval
```

Training uses the independent PyTorch implementation on CPU: seed 2026, two layers, four heads, dimension 64, MLP 256, context 32, and 3,000 final-answer supervised Adam steps. The original dimension-32 candidate plateaued near 61% accuracy; capacity and learning rate were adjusted before export. Train/validation/test token sequences are disjoint and determined by a recorded hash rule. Test inputs participated in the quality gate during development; they are not a pristine post-development audit.

The exporter saves 411,648 weight bytes, a hash-checked manifest/tokenizer, evaluation, three measured recipes, forward/gradient/Adam probes, and all 453 independent held-out logit fixtures. Training and recipe gates run before replacing public model assets. A selected-weight snapshot stays in `.cache/retrieval`; it does not contain optimizer resume state. See `docs/retrieval.md` for the exact task and limitations.

## Project status

- Implemented: scaffold and lockfiles, model contract, artifact validation, independent fixtures, full forward execution, gradients, selective baseline/modified inspection, worker probe, and an initial intervention screen.
- Implemented in Milestone C: a 32 MiB baseline cache with LRU eviction and read leases, exact same-layer residual deltas, suffix execution, full fallback, reset/identity reuse, and cached inspection. The screen shows the actual execution strategy and reuses unchanged baseline inspector slices during slider edits.
- Implemented in Milestone D: deterministic retrieval training, held-out and length evaluations, measured recipes, a trained default model, checkpoint switching, and responsive attention inspection.
- Verified: 41 engine tests, five dataset/reference/artifact tests, 319 direct cached/reference cases, and 26 production tests per browser in CI Chromium and Firefox, including 1,000 comparisons on each supported CPU/WebGL combination. The earlier 14 production tests also passed in Playwright WebKit on the recorded Mac host.
- Implemented in Milestone E: versioned experiment export/replay, exact numeric strength, attention-cell arithmetic, lazy projection/residual/MLP vectors, and CPU worker restart.
- Implemented in Milestone F: named-host browser latency measurements, source/build artifact auditing, pinned-action CI, verified Pages deployment, and hosted asset/workflow checks.
- [Live application](https://jtmcbride.github.io/transformer-scope/). CI passes Chromium CPU/WebGL, Firefox CPU (worker WebGL unavailable on that runner), and hosted release checks. Installed Safari passes manual CPU/WebGL reference checks and hosted CPU verification.
- Implemented in expansion (started before the educational gate by project decision): a residual-stream read-out (logit lens) and an input-embedding view, and an ablation explorer with pairwise non-additivity, all verified against PyTorch. The intervention panel can also redirect a head's attention (`attention.override`) and patch a residual vector from another input of the same length (`residual.replace`); see `docs/model-contract.md`.
- Story branching: select a generated token in either comparison path, inspect its exact computation, and make another head change. Earlier comparisons stay available through Back to previous comparison and the persistent branch map. The workspace saves locally and supports complete export/import, including paused generation. Edits can apply once or throughout continuation; all inherited head settings and sampling counters are preserved. See `docs/controlled-continuation.md`.
- TinyStories real text: the locally trained 48,000-step, 1.06M-parameter checkpoint is shipped as an experimental model with a 2,048-token BPE vocabulary and 64-token context. Its original name-copying gate remains failed; numerical verification and release behavior are documented in `docs/tinystories-deployment.md`.
- Implemented for release validation: an opt-in on-device harness (`validation.html`), a physical-device protocol, an educational evaluation protocol with a findings template, and WebGL context-loss detection with CPU recovery.
- Remaining release validation: educational sessions (the session page is ready), and performance across more named hardware/browser combinations. The physical lower-powered phone run is deferred by project decision and not validated.

See `PLAN.md`, `docs/model-contract.md`, `docs/validation.md`, and `docs/benchmarks.md` for scope, semantics, and measured evidence.

`npm run benchmark:browser` records isolated production cold-ready, warm UI, and full/cached RPC samples for both models/backends in Chromium and WebKit. See `docs/benchmarks.md` for scope, and `docs/deployment.md` for release checks.

`npm run analysis:heads` ablates each head on all 453 held-out inputs with the engine on CPU (about a minute) and writes `.cache/head-coverage.json`. The recorded result and its limitations are in `docs/retrieval.md`.

`npm run benchmark:cache` records an isolated Node CPU comparison for a three-layer fixture at early/middle/late edits. Its raw output goes to `.cache/cache-benchmark.json`. It measures forward computation and logit readback, not complete browser interaction latency.
