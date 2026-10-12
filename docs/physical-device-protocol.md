# Physical-device validation protocol

This protocol closes the Milestone F gate for "at least one physical lower-powered mobile device." It has two parts: the opt-in harness (automated measurements on the device) and a manual touch checklist. Both are required. No physical device has completed it yet; see `docs/validation.md`.

## Evidence classes

Record exactly one class per run. They are not interchangeable.

| Class | What it shows | What it does not show |
| --- | --- | --- |
| Viewport emulation (desktop browser at a phone width) | Layout and overflow at that width | Touch input, mobile browser engine, mobile CPU/GPU, memory limits |
| Playwright WebKit | WebKit engine behavior under automation | Installed Safari, iOS Safari, physical hardware |
| Installed desktop Safari | The shipped Safari engine on a Mac | iOS/iPadOS Safari, touch, mobile GPU or thermal limits |
| Automated browser (Playwright/CI) | Repeatable regressions on named runners | Any physical-device claim |
| **Physical mobile device** | Real touch, mobile browser, device CPU/GPU, memory, and thermal behavior | Other devices, browsers, or OS versions |

Only the last class satisfies this gate. Remote device farms count only if the device is physical and the session records its model and OS; record the provider in notes.

## Device selection

Prefer a lower-powered device: a mid-range Android phone at least three years old, or the oldest supported iPhone available. Record why it qualifies (for example, its release year or chipset). One device satisfies the first-release gate; additional devices are useful but do not replace it.

## Before starting

1. Charge above 50%. Record battery level, whether it is charging, and whether low-power/battery-saver mode is on. Run once in the device's normal state; low-power mode is an optional second run.
2. Close other tabs and apps. Let the device rest for five minutes if it is warm; record "cool" or "warm."
3. For a true cold load, clear site data for `jtmcbride.github.io` (iOS: Settings → Safari → Advanced → Website Data; Android Chrome: Site settings → All sites). Otherwise record "warm cache." The harness separately records whether `weights.bin` came from the network (`transferSize` > 0).
4. Hold the device in portrait orientation unless a step says otherwise. Keep the screen on and the page visible for the entire harness run.
5. For failures, connect remote debugging if available (Safari Web Inspector from a Mac, or `chrome://inspect` for Android Chrome) to capture console errors.

## Identification fields

Record these for every run. The harness records the browser-reported values automatically; the observer supplies the rest.

| Field | Source | Example |
| --- | --- | --- |
| Device manufacturer and model | Observer (Settings → About) | Google Pixel 6a |
| OS and version | Observer and harness `environment.userAgent`/`userAgentData` | Android 16 |
| Browser and exact version | Observer (browser About page) and harness | Chrome 141.0.7390.37 |
| Why it is lower-powered | Observer | 2022 mid-range, Tensor G1 |
| Battery %, charging, low-power mode, thermal state | Observer | 78%, unplugged, off, cool |
| Network and cache state | Observer | Wi-Fi, site data cleared |
| Deployed commit | Harness `release.data.commit` | full 40-character SHA |
| Model revisions | Harness `combinations[].modelRevision` | `d95b2853…` |
| Actual backend per run | Harness `combinations[].actualBackend` | `cpu` / `webgl` / unsupported |
| GPU renderer | Harness `environment.mainThreadWebgl.renderer` (main-thread context; may be masked) | Adreno (TM) 642L |
| Observer and date | Observer | name, UTC date |

## Part A — harness run

1. Open <https://jtmcbride.github.io/transformer-scope/validation.html> on the device. This page is not linked from the application.
2. Set **What is running this page?** to "Physical mobile device." Enter the device, browser, battery, and thermal state in the description field.
3. Keep the default scope: both checkpoints, CPU and WebGL, 5 warm-up and 40 measured edits per layer, interface measurement on. Press **Start validation** and keep the page in the foreground. Expect several minutes on slower devices.
4. When the result appears, press **Download JSON** (or **Copy JSON** if the browser cannot save files) and keep the file unedited.
5. Record the outcome. If `pageHiddenDuringRun` is true, the timings are invalid: repeat the run.

What the harness checks for each checkpoint/backend:

- `ui-cold-readiness`: the real application, loaded in an embedded frame, from navigation start until the first completed comparison appears.
- `ui-warm-interaction`: 40 strength edits per layer (head 1, all positions, strengths .25/.5/.75/1.5) from the input event until the completed result appears. Frame-opportunity time is reported separately; it adds two animation-frame callbacks and is not measured display latency. Edits are synthetic events and exclude touch hardware and OS input latency.
- `ui-baseline-reset` and `ui-cpu-recovery`: reset reuses the immutable baseline with no displayed change; restart creates a CPU worker and preserves the experiment.
- `worker-load`, `reference-probe`, `prepared-recipes`, `replay-roundtrip`, `resource-cleanup`: model identity, PyTorch forward/gradient/Adam parity, all three recipes through cached and full execution, in-memory export/replay, and tensor/cache release. These use the existing tolerances.
- `computation-rpc`: full and cached comparison round trips, with the worker-reported computation time recorded separately.

Statuses: `passed`; `failed` (a requirement was not met, with the reason); `unsupported` (the backend could not start on this device, such as WebGL without float32 rendering); `skipped` (not applicable, or a prerequisite was unsupported). CPU must never be unsupported. No timing threshold is applied.

## Part B — manual touch checklist

Use the application at <https://jtmcbride.github.io/transformer-scope/>. For each step record **pass**, **fail**, or **unsupported**, plus notes and a screenshot reference when the observation is visual.

| # | Action | Expected observation |
| --- | --- | --- |
| B1 | Open the application. | Status reaches "Showing 0.00×". The note reads "Ablation changes the answer". Backend shows "CPU worker". No horizontal page scroll. |
| B2 | Choose each **Prepared experiment** with the native picker. | *Ablation changes the answer*: the correct answer's modified bar drops from about 99.99% to near 0%. *A reachable edit has little effect*: bars look identical; the change is tiny but the path note is absent. *An earlier final-layer edit cannot reach the answer*: the note "This setting has no active computational path…" appears and Δ log p is 0. |
| B3 | Drag the **Head strength** slider with a finger, slowly and then quickly. | The status shows "Computing… previous …" while dragging and settles on the value under the thumb when released. Bars dim while pending and never show a value you did not request last. The page does not scroll while you drag. |
| B4 | Tap **Exact head strength**, type `0.37` on the on-screen keyboard, and press Enter/Done. Then try `2.5`. | 0.37 is applied ("Showing 0.37×"). 2.5 shows "Use a strength from 0 to 2." and leaves the result unchanged. |
| B5 | Tap token buttons to change the prediction position; use **Layer**, **Head**, and **Apply at** pickers. | Each tap or selection produces a new completed result. Targets are large enough to tap without zooming. |
| B6 | In the attention inspector, scroll the matrix horizontally with a finger. Tap an unmasked cell and a masked (·) cell. | The matrix scrolls inside its panel; the page does not. The cell panel shows Q · K, √d scaling, the computed score, and the softmax weight; a masked cell shows "-1e9 · future position". |
| B7 | Note one attention-cell weight. Change the strength to 0, then 2. | The attention weights do not change: head-output scaling acts after attention. The head-output vector's modified column changes. |
| B8 | Scroll the head-output vector list, then tap **Inspect projection, residual, and MLP**. Change stage, layer, and position. Scroll the 64-row table. | The table scrolls within its panel. Baseline, modified, and change columns update after each selection. |
| B9 | From an edited state, tap **Reset to baseline**. | "Showing 1.00×", every row shows +0.00 pp, and the strategy reads "Baseline reused". |
| B10 | Tap **Export experiment**, then **Import experiment** and choose the saved file. | The file is saved (iOS: Files or Downloads; Android: Downloads). Import shows "Replay matches saved results". If the browser cannot save or pick the file, record **unsupported** with the exact behavior. Do not count it as a pass. |
| B11 | Choose WebGL in **Backend**. Then tap **Restart computation worker on CPU**. | WebGL either loads or shows "WebGL worker unavailable…" (record which). Restart returns to "CPU worker" and keeps the same input and settings. |
| B12 | Switch to another app or lock the screen for one minute, then return. | The last result is still shown, or the page shows an error that **Restart computation worker on CPU** recovers. Record which, especially on WebGL (possible GPU context loss). |
| B13 | Rotate to landscape and back. | No horizontal page scroll; the experiment and result are preserved. |
| B14 | Tap **Verify against PyTorch**. | "Reference checks passed" with forward, gradient, and Adam errors, and a retained tensor change of 0. |

## Recording results

1. Save the harness JSON unchanged as `docs/device-results/<UTC date>-<device>-<browser>.json`.
2. Copy `docs/device-results/manual-checklist-template.md`, name it to match the JSON file, and fill it in.
3. Add a row to the "Physical-device results" section of `docs/validation.md`: device, OS, browser, deployed commit, actual backends, harness outcome, manual outcome, and links to both files. Report harness medians/p95 values with their exact definitions, separately from desktop results.

A failed check is a finding, not a reason to change tolerances, regenerate expected arrays, or rerun until it passes. Investigate it, fix the cause, deploy, and repeat the full protocol on the new commit.

## Gate criteria

The physical-mobile gate passes when one lower-powered physical device has:

- A harness run with outcome `passed`: every CPU check passes for both checkpoints, and WebGL either passes or is explicitly `unsupported`.
- Manual steps B1–B14 pass, except B10–B12, which may be explicitly `unsupported` with a recorded reason.
- Recorded timings. The 200 ms warm target was set for a named desktop configuration; report mobile results against it for information, not as a pass/fail threshold.
