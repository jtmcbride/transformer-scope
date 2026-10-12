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
- **Firefox 157.0:** the installed Playwright runtime exited before opening a page with `Could not find profile folder`. A retry under `/private/tmp` had the same result. Browser/model compatibility is unverified on Firefox; this is not a product test pass or a numerical test failure.

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
