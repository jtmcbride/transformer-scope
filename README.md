# Transformer Microscope

A browser laboratory for inspecting transformer computations and measuring internal interventions. The first implementation includes a verified two-layer transformer, a dedicated computation worker, and a head-scaling screen.

**The default checkpoint learns a controlled symbol retrieval task.** It answers 453/453 held-out three-binding inputs correctly. It is not a general language model; accuracy drops with different binding counts. The untrained numerical verification checkpoint remains selectable.

## Run

Use Node.js 24 and npm:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/transformer-scope/`. The trained input format is `A = 0 ; B = 3 ; C = 1 ; ? A =` with spaces around every symbol. Keys A–D are distinct; values 0–3 may repeat. Choose a prepared experiment or write your own input. The selected verification checkpoint accepts A–O with at most seven symbols. Choose a prediction position, layer, head, token scope, and head strength. The baseline remains immutable; factor 1 restores it. Export a completed experiment as JSON, then import it to restore settings and compare recomputed predictions with saved values. See `docs/experiment-format.md` for the versioned format.

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
- Verified: 41 engine tests, five dataset/reference/artifact tests, 319 direct cached/reference cases, and 21 production tests per browser in CI Chromium and Firefox, including 1,000 comparisons on each supported CPU/WebGL combination. The earlier 14 production tests also passed in Playwright WebKit on the recorded Mac host.
- Implemented in Milestone E: versioned experiment export/replay, exact numeric strength, attention-cell arithmetic, lazy projection/residual/MLP vectors, and CPU worker restart.
- Implemented in Milestone F: named-host browser latency measurements, source/build artifact auditing, pinned-action CI, verified Pages deployment, and hosted asset/workflow checks.
- [Live application](https://jtmcbride.github.io/transformer-scope/). CI passes Chromium CPU/WebGL, Firefox CPU (worker WebGL unavailable on that runner), and hosted release checks. Installed Safari passes manual CPU/WebGL reference checks and hosted CPU verification.
- Implemented for release validation: an opt-in on-device harness (`validation.html`), a physical-device protocol, an educational evaluation protocol with a findings template, and WebGL context-loss detection with CPU recovery.
- Remaining release validation: educational sessions (the session page is ready), and performance across more named hardware/browser combinations. The physical lower-powered phone run is deferred by project decision and not validated.

See `PLAN.md`, `docs/model-contract.md`, `docs/validation.md`, and `docs/benchmarks.md` for scope, semantics, and measured evidence.

`npm run benchmark:browser` records isolated production cold-ready, warm UI, and full/cached RPC samples for both models/backends in Chromium and WebKit. See `docs/benchmarks.md` for scope, and `docs/deployment.md` for release checks.

`npm run benchmark:cache` records an isolated Node CPU comparison for a three-layer fixture at early/middle/late edits. Its raw output goes to `.cache/cache-benchmark.json`. It measures forward computation and logit readback, not complete browser interaction latency.
