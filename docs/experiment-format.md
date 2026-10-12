# Portable experiments, versions 1 and 2

Export saves the completed experiment, including its executed token IDs and controls. Unsubmitted textarea drafts and pending edits are excluded. Import recomputes predictions against an installed checkpoint and compares every saved logit with the live result. It never substitutes saved probabilities for current computation.

The JSON format is `transformer-microscope-experiment`. Exports are written as **schema version 2**. Version 1 files, which record one head scaling, import unchanged. Files are limited to 1 MiB. The payload includes:

- Capture timestamp and model ID, architecture version, revision, weight hash, and tokenizer hash.
- Canonical space-separated input and exact token IDs including BOS.
- Prediction position, target token ID, and exactly one intervention.
- For a residual patch, `observation.source`: the source input's canonical text and token IDs.
- Complete baseline and modified vocabulary logits at the selected prediction position.
- Display top-K, temperature 1, natural-log metric units, JSD, maximum logit change, and target log-probability change.
- Computation backend and serialized float32 dtype, worker WebGL packing and float32-texture flags (null for CPU).

## Interventions a file may contain

A file holds only what the screen can restore. Anything else is refused rather than approximated.

| Kind | Version | Restorable form |
| --- | --- | --- |
| `head.scale` | 1 and 2 | One head at all positions or one explicit position. |
| `attention.override` | 2 | One head and query position, with weights that are one-hot on a source at or before the query, or exactly 1 on every position from 0 to the query (even). |
| `residual.replace` | 2 | One layer input or post-attention residual at one position, with the replacement values. It must record its `source` input, which is tokenized consistently and has the same length as the input. A `source` on any other kind is rejected. |

Semantics of each kind are in `docs/model-contract.md`.

## Validation and replay

Runtime run IDs and GPU tensors are deliberately absent. Before the current observation is replaced, import checks:

- checkpoint identity;
- text/token consistency of the input and, for patches, the source;
- intervention validity against the installed model's configuration (indices, lengths, value ranges, causality);
- consistency of the saved metrics with the full logit arrays.

Unknown schemas or models, stale weights or symbols, more than one intervention, and unrestorable forms are rejected. A saved installed model is selected automatically when needed.

A saved patch is replayed by first creating the source input's run in the current worker. The screen then patches the residual read from that run, so the replay recomputes the source vector as well. Replay uses `abs(current - saved) <= 2e-5 + 1e-5 * abs(saved)` on each baseline and modified logit. The screen reports both maximum errors and the saved and current backends. A self-consistent file whose measurements differ from new computation remains inspectable, with a discrepancy message. These bounds reuse the documented numerical verification tolerances. A matching file is a reproducibility check for those values, not proof of provenance or general model correctness.

Export and replay preserve non-final prediction positions and targets outside the top-K chart. Manual changes remove the replay report when the replacement result completes. Earlier releases of the application read only version 1 and refuse version 2 files.

Implementation: `src/engine/experiment.ts`.

- **Engine tests** cover round trips, version 1 compatibility, refusal of version 2 kinds in a version 1 file, and one-hot and even redirects with unrestorable rows rejected. They also cover patches with sources, rejecting missing, misplaced, short, or mismatched sources, wrong value widths, and multiple interventions, as well as complete-array metrics, invalid settings, size limits, and discrepancy reporting.
- **Browser tests** cover actual downloads and imports for head scaling, attention redirects, and residual patches. This includes restoring the mode and every control, re-creating a patch's source, switching checkpoints, CPU-to-WebGL replay, altered measurements, and rejected imports that preserve the previous result.
