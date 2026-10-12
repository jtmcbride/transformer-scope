# Transformer Microscope

A browser laboratory for inspecting transformer computations and measuring internal interventions. The first implementation includes a verified two-layer transformer, a dedicated computation worker, and a head-scaling screen.

**The included model is untrained.** It is a deterministic numerical verification checkpoint, not the planned associative-retrieval demo or a language model with validated behavior.

## Run

Use Node.js 24 and npm:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/transformer-scope/`. Input accepts space-separated A–O symbols, with at most seven symbols plus the beginning token. Choose a prediction position, layer, head, token scope, and head strength. The baseline remains immutable; factor 1 restores it.

The tiny verification model defaults to CPU to avoid GPU shader compilation overhead. WebGL can be selected explicitly; larger models attempt WebGL with CPU fallback. The verification button checks actual browser forward values, gradients, and an Adam update against exported PyTorch results.

## Check

```sh
npm test
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

## Project status

- Implemented: scaffold and lockfiles, model contract, artifact validation, independent fixtures, full forward execution, gradients, selective baseline/modified inspection, worker probe, and an initial intervention screen.
- Implemented in Milestone C: a 32 MiB baseline cache with LRU eviction and read leases, exact same-layer residual deltas, suffix execution, full fallback, reset/identity reuse, and cached inspection. The screen shows the actual execution strategy and reuses unchanged baseline inspector slices during slider edits.
- Verified: 29 engine tests, 319 direct cached/reference cases, and 10 production tests across Chromium and WebKit, including 1,000 comparisons on each CPU/WebGL combination.
- Next: train and evaluate the controlled-task checkpoint (Milestone D), finish the experiment UI, and verify deployment.
- Remaining release validation: actual Safari, physical mobile devices, checkpoint educational quality, and end-to-end performance on named hardware.

See `PLAN.md`, `docs/model-contract.md`, `docs/validation.md`, and `docs/benchmarks.md` for scope, semantics, and measured evidence.

`npm run benchmark:cache` records an isolated Node CPU comparison for a three-layer fixture at early/middle/late edits. Its raw output goes to `.cache/cache-benchmark.json`. It measures forward computation and logit readback, not complete browser interaction latency.
