# Browser compatibility and performance evidence

The first implementation exercises the built production worker on fixed independent PyTorch fixtures. Timing the complete verification probe includes additional captures, gradient computation, optimizer allocation/update, and kernel compilation. It is not interactive comparison latency.

## Measurement protocol

- Record exact browser user agent and active backend.
- Check CPU independently even when automatic backend selection chooses WebGL.
- Check explicit WebGL independently; record unsupported contexts rather than silently claiming accelerated execution.
- Capture the backend's float32 texture flag on WebGL and the logical retained-tensor delta after disposing probe resources.
- Keep numeric checks and time measurements separate. No latency pass/fail threshold is imposed on the verification probe.
- Production UI displays worker comparison time only. That value excludes browser event scheduling, message transit, additional inspection reads, and rendering; it is not an end-to-end response benchmark.

## Initial scope

The fixture model contains 1,408 parameters and a five-position probe input. This does not establish performance for the proposed larger explorer model or the planned trained checkpoint.

Suffix execution is implemented and has a separate Node CPU measurement below. Cold/warm browser response measurements on named physical devices and the larger trained checkpoint remain part of Milestone F.

## Automated results

Recorded on client date 2026-10-11, macOS 27.0.1, Playwright 1.64.0, TensorFlow.js 4.22.0. The exact browser user agents and raw measurements are preserved in `browser-probe-results.json`. GPU renderer/hardware identity was not captured; these are browser-backend compatibility results rather than named-GPU benchmarks.

- **Chromium 156.0.8078.4:** CPU and unpacked WebGL passed forward, all-gradient, and Adam-update parity. Maximum errors were respectively `1.19209e-6`, `3.57628e-7`, `5.58794e-7` on CPU, and `1.43051e-6`, `1.78814e-7`, `6.48201e-7` on WebGL. Probe durations were 19.7 ms and 4,874.3 ms.
- **Playwright WebKit 27.2:** CPU and unpacked WebGL passed the same checks. Maximum errors were respectively `1.19209e-6`, `3.57628e-7`, `5.58794e-7` on CPU, and `1.90735e-6`, `2.38419e-7`, `8.34465e-7` on WebGL. Probe durations were 23 ms and 1,101 ms.
- Both WebGL runs reported float32 textures and `WEBGL_PACK=false`. Every CPU/WebGL probe ended with a zero retained-tensor delta.
- The initial production suite passed eight tests across Chromium and WebKit, including interaction, mobile viewport, input recovery, and corrupt-artifact checks. Milestone C extends that suite to 10 tests with cached/full parity and resource stress. Screenshots were inspected for layout and numerical presentation.
- **Firefox 157.0:** the installed Playwright runtime exited before opening a page with `Could not find profile folder`. A retry under `/private/tmp` had the same result. That local Firefox runtime remains unverified; Linux CI now separately validates Firefox 157 CPU (see `docs/ci-browser-results.json`), with worker WebGL unsupported on that runner; this is not a product test pass or a numerical test failure.

The durations above are single-run observations for the full verification probe. They include model kernels, capture/readback, backward execution, and optimizer work, and may reflect shader compilation/cache state. They exclude preceding backend initialization. They cannot be used as steady-state comparison latency or as a CPU-versus-GPU performance conclusion.

The initial packed WebGL path produced non-finite tied-token-embedding gradients on WebKit even though forward parity passed. Unpacked WebGL resolved that failure without changing expected arrays or tolerances. CPU is the default for the 1,408-parameter verification model; enabling packing or choosing an accelerated default for a larger checkpoint requires new measurements.

Re-run `npm run build` followed by `npm run test:browser -- --project=chromium --project=webkit` to reproduce the currently runnable suite on this host. The full `npm run test:browser` also attempts Firefox. The machine-readable test report and probe attachments are written to `test-results/`.

## Cached suffix measurements

`cache-verification-results.json` records cached/full logit and residual comparisons, independent parity, and resource observations for CPU and unpacked WebGL in both Chromium and WebKit. All four combinations pass 14 cached comparisons plus 1,000 sequential edits. The maximum cached/full logit difference across the recorded browser checks is about `1.43e-6`; tensor counts and the 640-byte cache payload remain stable.

`cache-benchmark-results.json` records a separate Node CPU run on darwin/arm64, Node 24.18.0, TensorFlow.js 4.22.0, using the three-layer seed-1 model (D=8, H=2, MLP=16, V=16), eight input tokens, and head 0 scaled to 0.5 at all positions. Each layer uses 10 warm-up pairs and 80 measured pairs, alternating full/cached order. The command ran after the browser workloads completed.

- Layer 1: full median 0.845 ms / p95 1.171 ms; cached median 0.704 ms / p95 0.929 ms; median ratio about 1.20×.
- Layer 2: full median 0.779 ms / p95 0.896 ms; cached median 0.391 ms / p95 0.611 ms; median ratio about 2.00×.
- Layer 3: full median 0.761 ms / p95 0.964 ms; cached median 0.159 ms / p95 0.215 ms; median ratio about 4.79×.

These measurements cover forward computation and all-logit readback. They exclude worker transit, comparison metrics, subsequent inspection, rendering, and disposal time. They demonstrate reuse on this fixture and runtime; they do not establish browser interaction latency, GPU speedup, or performance of the proposed larger model. The benchmark records observations without imposing a latency/speedup threshold.

Run `npm run benchmark:cache` to regenerate `.cache/cache-benchmark.json`. Review and copy a measurement into the permanent docs only when its scope and environment are recorded.

## Named-host browser latency

Recorded 2026-10-12 UTC on MacBookPro18,1, Apple M1 Pro, 10 logical CPU cores, 16 GiB RAM, macOS 27.0.1; Node 24.18.0, Playwright 1.64.0, TF.js 4.22.0. Raw samples, exact user agents, checkpoint revisions, shapes, and backend precision are in `browser-latency-results.json`. Benchmarks run sequentially in headless Chromium/WebKit, separate from regression workloads. They are observations, with no timing threshold in tests.

Each browser/backend/model uses a fresh page context. Cold-ready time runs from navigation timing origin until the initial completed comparison and its default inspectors commit. HTTP/model caches belong to that fresh context; OS/filesystem/driver/shader caches are not forcibly reset. One cold-ready observation per combination is reported, not a cold-load distribution.

Warm UI edits use head 0 at all positions, alternating .25/.5/.75/1.5 strength, independently at the first and final layers. Five warm-up edits precede 40 samples per layer. An in-page MutationObserver measures event dispatch to the completed result's DOM commit, including scheduling, worker RPC, comparison, default vector inspection/readback, and React updates. Two animation-frame callbacks then provide a frame-opportunity measurement; this includes an intervening paint opportunity, not instrumentation of physical display pixels. Additional stage/cell inspectors remain collapsed. These are sequential edits, not a claim about burst-update responsiveness.

For the trained 102,912-parameter, 16-token checkpoint:

- Chromium CPU: cold-ready 155 ms; DOM-commit median/p95 7.8/8.2 ms at layer 1 and 3.9/4.6 ms at layer 2. Frame-opportunity p95 34.0/33.6 ms.
- Chromium unpacked WebGL: cold-ready 2,114 ms; DOM-commit median/p95 18.7/24.3 ms and 12.8/17.0 ms. Frame-opportunity p95 50.1/34.1 ms.
- WebKit CPU: cold-ready 226 ms; DOM-commit median/p95 42/44 ms and 12/13 ms. Frame-opportunity p95 68/36 ms.
- WebKit unpacked WebGL: cold-ready 324 ms; DOM-commit median/p95 23/25 ms and 16/18 ms. Frame-opportunity p95 52/50 ms.

The smaller verification model's frame-opportunity p95 ranges from 33.4 to 52 ms; its cold-ready observations range from 204 to 2,186 ms. Both models/backends stay below the initial 200 ms warm target on this host in the measured settings. This does not establish physical-mobile or installed-Safari performance.

A separate fresh worker measures its first baseline RPC, then five warm-up pairs and 40 alternating-order full/cached comparison RPC pairs per layer, at factor .5. RPC latency includes transit, forward/logit readback, metrics, and response. It excludes subsequent inspector reads and rendering; releasing each modified handle occurs after the timed response. For the trained model, full/cached RPC medians are:

- Chromium CPU: 6.7/5.3 ms at layer 1; 6.5/2.1 ms at layer 2.
- Chromium WebGL: 16.15/11.6 ms; 16.0/5.9 ms.
- WebKit CPU: 31/25 ms; 29/10 ms.
- WebKit WebGL: 11/9 ms; 11/8 ms.

WebKit timer resolution yields integer-millisecond readings, including occasional 0 ms on the tiny late-layer fixture. These values do not imply instantaneous computation. Whole-pipeline CPU/WebGL differences depend on the browser and shape; do not infer a universal acceleration ratio.

Automatic loading now chooses CPU for this exact trained revision and the tiny verification models. The measured trained checkpoint meets the warm target on CPU and avoids Chromium's GPU compilation startup cost. WebGL remains selectable and can be faster on the measured WebKit workload. A changed trained revision requires recalibration; other larger artifacts retain the prior WebGL-attempt/CPU-fallback policy.

Run `npm run build`, then `npm run benchmark:browser`. Raw output is `.cache/browser-benchmark.json`. Review hardware/configuration and scope before promoting a result to documentation.
