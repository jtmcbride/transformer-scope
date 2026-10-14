# Static build and release verification

The application requires static file hosting with JavaScript module workers and JSON/binary model assets. There is no inference server. Vite currently builds for the `/transformer-scope/` subpath; preserve that path on the host or change `base` and repeat production tests before deployment.

## Build and inspect

```
npm ci
npm test
npm run build
npm run check:artifacts
npm run test:browser -- --project=chromium --project=webkit
npm run preview -- --port 4174
```

`dist/` is the deployment artifact. Serve its contents under `/transformer-scope/`, including `assets/`, `models/`, the opt-in `validation.html` diagnostic page, and the opt-in `study.html` session page. Neither page is linked from the application. Serve worker JavaScript as JavaScript, JSON as JSON, and `weights.bin` as binary data. Use HTTPS (or localhost for development), so Web Crypto checksums and worker backends are available. Source files, Python environments, fixture generators, and `.cache/` are not deployment inputs.

The artifact checker independently verifies canonical manifest revisions, exact binary/tokenizer hashes, tensor byte layout, finite weights, tokenizer dimensions, and retrieval metadata revisions in both source and built assets. Production browser tests exercise the actual subpath and emitted worker rather than development imports.

## Continuous verification

`.github/workflows/ci.yml` runs on main pushes, pull requests, and manual dispatch, with read-only verification permissions and action commit pins from official releases. It installs Node 24.18.0 and locked npm dependencies, tests standard-library dataset invariants and artifact-corruption gates, runs all engine tests, builds and audits assets, and runs production Chromium and Firefox tests. It retains the static build and browser diagnostics for seven days. It does not train new weights or regenerate numerical expectations.

The first remote verification passed on Ubuntu 24.04 with Chromium at commit `00e814e`: [run 38254063282](https://github.com/jtmcbride/transformer-scope/actions/runs/38254063282). The extended Chromium/Firefox run also passed at `6a41a09`: [run 38254919942](https://github.com/jtmcbride/transformer-scope/actions/runs/38254919942). Chromium passes CPU/WebGL; Firefox 157 passes CPU and explicitly reports worker WebGL unavailable on the Linux runner. Exact reports are saved in `docs/ci-browser-results.json`. Local Chromium/WebKit checks remain separate evidence. After verification succeeds on main, a separate job deploys the identical static artifact to GitHub Pages. Only that job receives Pages write and OIDC permissions; pull requests never deploy. A final job checks the actual HTTPS site and preserves hosted diagnostics. Action references: [checkout](https://github.com/actions/checkout), [setup-node](https://github.com/actions/setup-node), [upload-artifact](https://github.com/actions/upload-artifact).

## Hosted release gate

For GitHub Pages, the repository subpath is already configured in Vite. The repository uses workflow-based Pages deployment at https://jtmcbride.github.io/transformer-scope/ with HTTPS enforced. Every build writes `release.json` containing the source commit, dirty-tree marker, model revisions, and SHA-256/byte length for every deployment asset. CI checks the deployed commit before exercising the site. Run the same check manually:

```
HOSTED_BASE_URL=https://jtmcbride.github.io/transformer-scope/ EXPECTED_COMMIT=<full-commit-sha> npm run test:hosted
```

The hosted tests verify:

- The page, module worker, all three manifests/tokenizers/binaries, recipes, evaluation, and probes return the intended content and MIME types under the subpath.
- The default TinyStories workspace runs real English text and compares every exported logit with the independent fixture. Its four-layer model is labeled experimental; see `tinystories-deployment.md` for quality results and limitations. The 47.3 MB numerical probe is fetched only on demand, while ordinary inference loads the 4.25 MB weights and tokenizer.
- Hash validation, prepared experiments, inspection, reset, export/replay, and CPU worker restart behave as in the production tests.
- Explicit CPU mode and automatic WebGL fallback remain usable. `?backend=cpu` selects CPU at initial load; `?backend=webgl` explicitly requests WebGL.
- The opt-in `validation.html` harness identifies the exact deployed commit and passes every supported check (Chromium; WebGL may be reported unsupported, never failed).
- Record real Safari/Firefox and a physical lower-powered mobile device, including browser/OS/backend and end-to-end timings, using `validation.html` and `docs/physical-device-protocol.md`. Viewport emulation and Playwright WebKit do not fulfill those gates.

Commit `29b6ade` added `validation.html` and a third hosted check. All three passed on the live site in [run 38263433282](https://github.com/jtmcbride/transformer-scope/actions/runs/38263433282). The first verified deployment passed at `71a5694`: [run 38258414801](https://github.com/jtmcbride/transformer-scope/actions/runs/38258414801). Exact live asset and workflow records are in `docs/hosted-release-results.json`; installed Safari smoke evidence is in `docs/native-safari-results.json`. The remaining release gates include a physical lower-powered mobile device and broader educational evaluation. Local performance results are in `docs/browser-latency-results.json` and `docs/benchmarks.md`; do not present them as every-device guarantees.

Deployment actions are pinned: [configure-pages](https://github.com/actions/configure-pages), [upload-pages-artifact](https://github.com/actions/upload-pages-artifact), and [deploy-pages](https://github.com/actions/deploy-pages). Reverting main to a previously verified source revision and letting the workflow rebuild/redeploy provides rollback; a failed post-deployment check reports the failure and does not silently claim the previous site is still active.
