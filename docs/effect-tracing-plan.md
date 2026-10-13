**Milestone 2 — Trace the effect**

Status: implemented. Attention-redirect tracing has since been added; see the next slice in `docs/frontend-workspace.md`. Builds on the first computation-workspace milestone in `docs/frontend-workspace.md`. The plan below is retained as the design and acceptance contract; implementation details and verification are in `docs/frontend-workspace.md` and `docs/computation-inspection.md`.

**Outcome**

After changing a head, a user can follow the completed intervention through the graph, see which vectors changed, and inspect where the residual readout's preferred token changes. The prediction comparison stays visible throughout.

The initial scope is one `head.scale` intervention on either shipped checkpoint: any supported factor, all positions or one position, and any prediction position. Tracing covers identity settings, a large effect, a small reachable effect, and an edit with no path to the selected prediction. Existing attention redirect and residual patch workflows remain available; their graph-based trace routes belong to the following milestone.

**User flow**

1. Run an input and adjust a head. Press **Follow the change** once the comparison completes.
2. Show the intervention's layer, head, strength, affected positions, and prediction position in a compact trace header. For a single-position edit, default inspection to that position. For all-position edits, default to the prediction position and keep **All positions edited** visible.
3. Overlay the possible route and measured node changes on the existing graph. Display the inspection position on every vector summary; give prediction-position readouts their own label.
4. **Previous** and **Next** walk through head output, shared projection, attention residual, MLP contribution, block output, later attention layers, and final readout. A step selects its graph node, expands its layer, and brings it into view. Manual graph selection synchronizes the step when the node is part of the trace.
5. Show a diagnostic token readout at residual stages and the actual prediction at the final stage. Keep the persistent prediction card visible.
6. **Exit trace** returns to ordinary graph inspection. Resetting the intervention produces a baseline/identity trace. Input changes, imports, checkpoint changes, and worker restarts reset trace navigation to the new completed experiment.

For single-position edits, show the original position and prediction position together. When they differ, display an explicit handoff at a subsequent attention layer: **Information at position 0 can enter attention at position 15**. An earlier-position edit in the final layer has local downstream changes but no route to a later prediction. The trace should show that local computation and explain where the route ends.

**Two independent graph annotations**

| Annotation | Source | Presentation |
| --- | --- | --- |
| Edited here | Completed intervention | Existing amber marker with layer, head, and position scope. |
| Can affect this prediction | Architecture and intervention positions | Connected route with a text label. |
| Local downstream computation | Architecture, when it cannot reach the prediction | Local route ending before the selected prediction. |
| Exact match | Measured baseline and modified arrays | Quiet **Exact match** badge. |
| Small measured difference | Nonzero differences within the existing numerical comparison bounds | **Small difference** badge with the nonzero value available. |
| Measured difference | Differences exceeding those bounds | **Changed** badge and compact magnitude. |
| Not measured / loading | Trace request state | Explicit state; never use the exact-match style. |

Selection keeps its existing outline, independently of these annotations. Use text and line styles as well as color. Report maximum absolute component difference and L2 difference; retain baseline and modified norms in the inspector. The small-difference threshold is a display convention, not a declaration that an effect is absent or unimportant. Use the project's `2e-5 + 1e-5 * abs(baseline)` per-component bound, retain exact measurements, and show scientific notation for nonzero values that would round to zero.

The graph represents possible data flow. Attention weights and vector-difference magnitudes do not determine whether an edge exists and are not causal-attribution scores. All heads remain visible as parallel branches. A guided sequence is an overview of those branches, rather than proof of a unique mechanism.

**Implementation sequence**

1. **Define topology and trace types.** Add `src/engine/trace-topology.ts` and `src/engine/trace.ts`. Use node identities containing activation stage, layer, position, and head where applicable. Map those identities to the existing graph nodes without making the engine depend on React components.

   Build the topology from the loaded configuration. Normalization, projections, residual additions, and MLPs preserve position. Attention can read sources at or before its query. Head scaling starts at `attention.head.modified`, after attention weights and raw head outputs. Construct the union of downstream routes from all edited positions and intersect it with the selected prediction's ancestor route. Keep local downstream nodes for the no-path case. Factor 1 has no effective edit. Root-to-prediction reachability must agree with `canReachPrediction`.

2. **Add one batched worker measurement.** Add `RunEngine.inspectEffectTrace` and a `TRACE` command in `src/workers/protocol.ts` and `model.worker.ts`.

   The request contains `baselineRunId`, `modifiedRunId`, and `predictionPosition`. Validate that both descriptors belong to the current session, have identical inputs, and that the modified descriptor refers to that baseline and contains exactly one supported head scaling. Reject stale or released runs using the existing engine errors.

   Preserve the comparison's selected-position logit array in the modified run descriptor as an internal snapshot. It is already available when `compare` computes the result; this allows trace measurements to be checked against the precise completed comparison without changing the experiment file format.

   Run at most one baseline forward and one modified forward, capturing the stages needed for the trace together. Compute compact summaries for all input positions in the worker: embedding, block inputs, normalization for attention, per-head weights/raw/scaled outputs, shared projection, attention residual, MLP contribution, block output, final normalization, and logits. Summarize attention by head and query row. Reuse captured residual rows for diagnostic readouts at the prediction position, rather than calling `LENS` and repeating forward passes.

   Return model revision, runtime generation, backend, run IDs, prediction position, intervention, per-position summaries, and complete baseline/modified diagnostic readout logits. Detailed vectors can still be requested through `INSPECT` when opened. The existing single-head UI can show separate attention-weight and output-change measurements.

   Full trace execution is a verified recomputation. Its final logits must agree with the baseline's recorded logits and the modified comparison snapshot within the existing forward tolerances. Return a trace error if that check fails; retain the completed comparison in the result card. The final prediction display continues to use the completed comparison, while intermediate readouts use the trace's diagnostic values.

   Check an estimated 8 MiB capture budget before allocating tensors. Return compact summaries instead of whole activation tensors. Release captures, readout tensors, and temporary tensors in `finally` blocks; do not add a persistent GPU trace cache or new run descriptors.

3. **Manage trace state independently of edits.** Add `src/hooks/useEffectTrace.ts`, driven by the completed display in `App.tsx`. Its key includes runtime generation, baseline/modified IDs, and prediction position. Keep open/closed state, loading/error state, report, inspection position, and current step separate from head controls.

   When closed, tracing adds no worker requests. When open, coalesce rapid completed edits, allow at most one trace request in flight with one latest queued request, and discard late responses by key. Never publish an old report under a new intervention. During an edit, keep the prior report explicitly labeled as the last completed trace, or show a loading state. Retain only the current report in frontend memory. Stepping, changing the inspected position, and selecting a measured node reuse that report without running the model again.

4. **Connect the graph and inspector.** Extend `ComputationGraph` with route and measurement annotations, controlled expansion for guided steps, and trace position labels. Update `NodeInspector` to consume matching trace summaries first and fetch full vectors only when requested. Share metric formatting and node mapping between the graph and inspector.

   Add an explicit cross-position connector at the relevant later attention layer. For all-position interventions, highlight the union of valid routes and label the inspected position; do not imply that one displayed position accounts for the full effect. Collapsed layers retain a trace indicator so the route remains understandable.

5. **Add guided controls and residual readouts.** Add `src/components/EffectTrace.tsx` with **Follow the change**, **Previous**, **Next**, an inspection-position selector, a step counter, and **Exit trace**. Add compact residual readout badges showing baseline/modified preferred tokens and target probabilities. Reuse `logSoftmax` and the existing tie rule.

   Intermediate rows say **Diagnostic readout**. Only the final row says **Model prediction**. Describe a flip as **The modified residual readout first prefers 0 here**, rather than assigning that layer a semantic role. Preserve parallel branches and residual bypasses in the graph while navigating.

   Desktop keeps the trace controls beside the graph. Mobile uses a compact step bar and the existing pinned prediction summary. Navigation remains keyboard-operable, announces the selected step, honors reduced motion, and keeps focused controls clear of the pinned card.

6. **Verify and document.** Extend the independent PyTorch export using shipped weight artifacts, including all three prepared retrieval recipes and selected verification cases. Add engine and production-browser trace tests, then document the trace semantics and measurement cost in `docs/frontend-workspace.md` and `docs/computation-inspection.md`.

**Acceptance cases**

| Case | Required result |
| --- | --- |
| Large-effect recipe: Layer 1 Head 4, strength 0 | Same-layer attention weights and raw head outputs match the baseline; the scaled head output is zero. Downstream vectors change. The modified residual readout still prefers 3 at Layer 1 block output and first prefers 0 after Layer 2 attention. Final logits agree with the completed comparison. |
| Small reachable effect: Layer 1 Head 2 at position 0 | A route from position 0 through Layer 2 attention to prediction position 15 is shown. Tiny measured differences remain visible numerically and are not converted to exact matches. |
| No path: Layer 2 Head 1 at position 0 | Local head/projection/residual/MLP changes can be inspected at position 0. There is no route to prediction position 15; that prediction is preserved. |
| Factor 1 / reset | No effective edit route. Baseline and modified measurements match, and no stale changed-node badges remain. |
| Edit after the prediction position | Causal masking excludes a route backward to that prediction. |
| Same-position final-layer edit | A direct route through projection, residual, MLP, and final readout is shown. |
| Navigation and inspected-position changes | Completed logits, intervention, export, and run IDs remain unchanged; no additional trace forward pass is issued. |
| Rapid edits, input replacement, checkpoint switch, import, restart, context loss | Obsolete reports cannot update the current graph. Loading and ended-session states remain explicit. |
| CPU / supported worker WebGL | Reference comparisons use the existing tolerances. Repeated tracing returns tensor counts and cache usage to their pre-trace values. |
| Mobile and keyboard | The prediction summary stays visible at 360, 390, and 768 px widths; navigation has visible focus and no page overflow or covered controls. |

**Release checks**

Run the new topology and trace engine tests, the existing engine suite, the production build, and the browser suite with the new trace interactions. Record which browser/backend combinations ran. Check cleanup after repeated traces and capture request counts to verify one batched measurement per completed trace and zero recomputation during navigation. Measure trace readiness separately from ordinary slider latency; the closed tracing UI must preserve the first milestone's interaction behavior.

The milestone is complete when all acceptance cases pass and a user can explain the large-effect, small-effect, and no-path examples by following the graph and reading the displayed measurements.
