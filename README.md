# Transformer Microscope

A browser laboratory for inspecting transformer computations and measuring internal interventions. The first implementation includes a verified two-layer transformer, a dedicated computation worker, and a head-scaling screen.

**The default checkpoint learns a controlled symbol retrieval task.** It answers 453/453 held-out three-binding inputs correctly. It is not a general language model; accuracy drops with different binding counts. The untrained numerical verification checkpoint remains selectable.

## Run

Use Node.js 24 and npm:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/transformer-scope/`. The trained input format is `A = 0 ; B = 3 ; C = 1 ; ? A =` with spaces around every symbol. Keys A–D are distinct; values 0–3 may repeat. Choose a prepared experiment or write your own input. The selected verification checkpoint accepts A–O with at most seven symbols. Choose a prediction position, layer, head, token scope, and head strength. The baseline remains immutable; factor 1 restores it.

The trained checkpoint attempts WebGL with CPU fallback. The tiny verification model defaults to CPU to avoid GPU shader compilation overhead. WebGL can be selected explicitly; larger models attempt WebGL with CPU fallback. The verification button checks actual browser forward values, gradients, and an Adam update against exported PyTorch results.

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
- Verified: 33 engine tests, two dataset/reference tests, 319 direct cached/reference cases, and 14 production tests across Chromium and WebKit, including 1,000 comparisons on each CPU/WebGL combination.
- Next: finish experiment provenance/export and computation traces (Milestone E), then verify deployment.
- Remaining release validation: actual Safari, physical mobile devices, wider educational evaluation, and end-to-end performance on named hardware.

See `PLAN.md`, `docs/model-contract.md`, `docs/validation.md`, and `docs/benchmarks.md` for scope, semantics, and measured evidence.

`npm run benchmark:cache` records an isolated Node CPU comparison for a three-layer fixture at early/middle/late edits. Its raw output goes to `.cache/cache-benchmark.json`. It measures forward computation and logit readback, not complete browser interaction latency.
