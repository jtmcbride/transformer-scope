# Controlled text continuation

The TinyStories workspace offers **Watch a story grow token by token** below the flow diagram, linked from the prompt editor. Step one token computes a single step and pauses; Play story continues at a readable pace, with 300 ms between completed steps. Stop generation immediately terminates the continuation worker and keeps the last completed step. Step/Play resume that exact prefix. Reset story returns to the completed prompt and captures fresh settings on the next step. Prompt drafts do not change a continuation. Running a new prompt, changing models, or restarting the diagram worker discards the old continuation.

The original prompt is muted in the reading view, generated text is darker, and numbered chips retain each generated token. Tooltips and the latest-token line show the raw model softmax probability, before temperature/top-p filtering. This is a short continuation from an experimental checkpoint, not a promise of a complete or coherent story. The earlier name-copying gate remains failed.

## Sampling contract

- Most likely token uses temperature zero, ranking eligible logits with lower token ID as the tie breaker.
- Seeded sampling uses temperature in (0, 2], top-p in (0, 1], and an unsigned 32-bit seed. Settings freeze at the first step. Reset to change them.
- BOS (0) and PAD (2) are excluded from generation. EOS (1) remains eligible and ends its branch. Probabilities reported on screen remain from the unfiltered full-vocabulary model softmax.
- Temperature rescales logits after subtracting the eligible maximum. Top-p sorts by probability with ID tie breaking, retains the smallest prefix whose cumulative probability reaches the threshold, includes the boundary token, and renormalizes that prefix before drawing.
- `drawAt(seed, step)` provides a counter-based uniform draw. Both branches receive the same draw at a given step. Their prompts can diverge, so this is controlled branching, not an assertion that later probability differences have the same input. Repeating settings on the same CPU runtime is reproducible; floating-point differences can change near-tied choices on other environments.
- Exact token IDs are appended to each prefix. Displayed text is never re-tokenized, so BPE merges, spaces, punctuation, and partial UTF-8 pieces cannot change computation boundaries. Preview holds an incomplete trailing character until further bytes arrive; invalid completed byte sequences display as replacement characters. Raw byte-piece labels remain in the token strip and download, including when a limit ends mid-character. Strict decoding of ordinary input remains strict and preserves a leading UTF-8 BOM.

Each branch stops independently at EOS, the new-token budget (1–64), or 64 total tokens including BOS. A full prompt cannot append a token. No rolling window, position rebasing, silent truncation, or KV cache is used. The default budget is 24 new tokens, further bounded by available context. The final appended token is included in that total-token limit.

## Original and edited stories

First make a head-output edit in the diagram, applied at all positions. Enable **Compare with current head edit** before starting. The panel snapshots that one head's layer, head, factor, and all-position scope. It applies the edit at every position of each growing edited prefix. The original branch uses unmodified weights. Later diagram edits do not change a started continuation; its captured edit stays labeled until Reset. Attention redirects, residual copying, and position-specific head edits remain available in the diagram but are not continuation controls in this slice.

A paired step commits only after both active branches finish. A cancellation or error cannot publish half a pair. Each branch then feeds its own chosen token into its own next step. A branch that has ended remains fixed while the other continues.

## Runtime and records

Generation loads the exact shipped checkpoint lazily into one separate CPU module worker, independently of the diagram's selected backend and cached runs. This avoids compiling a new WebGL shape for every prefix length and makes cancellation immediate without ending the inspection workspace. All temporary baseline and modified run descriptors/tensors are released after each prediction. Stop, Reset, model/prompt changes, and unmount terminate the continuation worker. There is at most one generation step in flight; it does not prefetch future tokens. The ordinary worker's numerical tolerances are unchanged.

Download continuation writes `transformer-microscope-continuation`, schema version 1: checkpoint identity/hashes, CPU backend, exact prompt text and IDs, captured sampling settings/edit, step count, both token prefixes, selected IDs/raw model probabilities, and per-branch stop reason. This is a generation record, separate from the replayable diagram experiment format. Continuation import, arbitrary historical-step inspection, rolling contexts, and generation performance claims on physical mobile are deferred.

## Verification

`scripts/export_continuation.py` independently rolls out the shipped weights in PyTorch for 16 original and 16 edited greedy steps from the red-ball prompt. The checked-in fixture fixes each chosen ID, probability, and complete prefix. Engine and production browser tests compare every generated token to it. The edited branch diverges after five matching tokens: the original continues “He ran to his friend” while the edited branch continues “He asked his friend.” This is one measured example, not a general role assigned to the head.

Additional engine checks cover BOS/PAD masking, eligible EOS, stable ties, temperature shift invariance, top-p boundary inclusion, seeded draws, atomic failures, independently ended branches, exact noncanonical BPE prefixes, incomplete UTF-8 and BOM preservation, context/token limits, and zero retained tensors after a complete paired rollout. Browser checks cover stepping, frozen edits despite diagram changes, view switching, seeded replay across step/play, draft separation, full-context refusal, mobile containment, cancellation during delayed inference, and model switching without late results. Hosted verification exercises paired generation against the same independent fixture after deployment.
