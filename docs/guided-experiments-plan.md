# Guided experiment launcher

Status: implemented and verified. Builds on the completed head, attention, and residual-copy tracing slices.

## Outcome

A new user can choose a prepared example, identify its input and edited head, inspect the measured result, and start following the computation without discovering controls below the workspace.

## Scope and sequence

1. Show the three installed retrieval recipes as visible cards: answer-changing ablation, small reachable effect, and a local edit with no route to the prediction. Use their validated metadata for titles and settings. Starting a card uses the existing baseline/comparison path and replaces input drafts explicitly. Keep the existing prepared-experiment selector and the default example.
2. Add a guide near the flowchart for the completed example. Show the edit's layer/head, strength, scope, and prediction position; display the actual completed top tokens, target probabilities, and nonzero Δ log p. Distinguish the architectural route from the measured effect. Explain that these are selected examples.
3. Provide **Inspect the edited head** and **Trace this example**. Inspection is navigation. Tracing remains opt-in, uses the existing queue, and returns to the first step when already open without making another trace request.
4. Carry recipe identity with each queued job and completed display. Cards and guidance identify the completed recipe, separately from a requested launch. Draft input/source changes retain the completed guide with a draft label. Applied custom interventions, reset, new inputs, replay, and worker restart show a custom experiment rather than falsely claiming the recipe is still active. Pending or ended-session guidance is explicitly labeled and its actions disabled.
5. Add production browser checks and documentation. No model, worker command, recipe artifact, or experiment-format change is needed.

## Acceptance

- Each card recomputes its recipe and agrees with the existing independent reference measurements; its selected head and position scope are visible.
- Inspecting the edited head preserves settings and adds no comparison or trace request. Opening a guide's trace uses one batched trace; returning to its start reuses that report.
- The large example exposes its diagnostic flip. The small example retains tiny nonzero values. The no-path example shows local changes and an unchanged final prediction.
- Draft edits, delayed comparisons, reset, export/replay, checkpoint switching, and restart cannot mislabel the completed example.
- Cards and guide actions work by keyboard at 360, 390, and 768 px, with no page overflow and the existing prediction/trace controls visible.
- Run the production build, engine suite, and Chromium browser suite; record tested backends and browser limits.

## Verification

The production build, 72 engine tests, and all 50 Chromium production browser tests passed. The three new launcher tests compare displayed recipe results with independent PyTorch trace fixtures, verify comparison/trace request counts, exercise drafts and delayed comparisons, export/replay/restart and checkpoint switching, and cover keyboard/layout behavior at 360, 390, and 768 px. The existing suite also verifies all 23 trace fixtures on CPU/WebGL and labels the guide after context loss. Firefox, WebKit, and physical devices were not tested for this slice.
