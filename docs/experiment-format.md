# Portable experiments, version 1

Export saves the completed experiment, including its executed token IDs and controls. Unsubmitted textarea drafts and pending slider edits are excluded. Import recomputes predictions against an installed checkpoint and compares every saved logit with the live result; it never substitutes saved probabilities for current computation.

The JSON format is `transformer-microscope-experiment`, schema version 1. Files are limited to 1 MiB. The payload includes:

- Capture timestamp and model ID, architecture version, revision, weight hash, and tokenizer hash.
- Canonical space-separated input and exact token IDs including BOS.
- Prediction position, target token ID, and one head-scaling intervention, at all positions or one explicit position.
- Complete baseline and modified vocabulary logits at the selected prediction position.
- Display top-K, temperature 1, natural-log metric units, JSD, maximum logit change, and target log-probability change.
- Computation backend and serialized float32 dtype, worker WebGL packing and float32-texture flags (null for CPU).

Runtime run IDs and GPU tensors are deliberately absent. Checkpoint identity and text/token consistency are validated before replacing the current observation. Unknown schemas/models, stale weights or symbols, unsupported multi-position edits, invalid settings, non-finite values, and metrics inconsistent with full logit arrays are rejected. A saved installed model is selected automatically when needed.

Replay uses `abs(current - saved) <= 2e-5 + 1e-5 * abs(saved)` on each baseline and modified logit. The screen reports both maximum errors and saved/current backends. A self-consistent file whose measurements differ from new computation remains inspectable, with a discrepancy message. These bounds reuse the documented numerical verification tolerances; a matching file is a reproducibility check for those values, not proof of provenance or general model correctness.

Export/replay preserve non-final prediction positions and targets outside the top-K chart. Manual changes remove the replay report when the replacement result completes. Files currently support the two installed checkpoints and the screen's one-head intervention controls; future formats must explicitly version broader interventions.

Implementation: `src/engine/experiment.ts`. Browser tests cover actual downloads, ignored input drafts, checkpoint switching, CPU-to-WebGL replay, altered measurements, and rejected imports that preserve the previous result. Engine tests cover round trips, complete-array metrics, invalid settings, version/size checks, and numerical discrepancy reporting.
