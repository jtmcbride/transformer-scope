# Educational evaluation protocol

This protocol tests the first release's educational gate with the three shipped recipes. It is a small formative study: it finds misunderstandings and interface problems. It does not estimate learning outcomes for a population. No sessions have been run yet. Record results in a copy of `docs/educational-findings-template.md`. Do not fill the template with hypothetical answers.

## Questions

1. **Prediction:** Before seeing a result, can participants predict the direction and rough size of a head intervention's effect on the selected prediction, and explain their reasoning?
2. **Attention invariance:** Do participants understand that scaling a head's output leaves that head's attention weights unchanged, because scaling happens after attention-weighted values are combined and before output projection?
3. **Propagation:** Do participants understand that the edited head output changes the residual stream, then the MLP, later layers, and the logits, and that it reaches later positions only through later attention layers?
4. **Small effect versus no path:** Can participants distinguish a reachable edit with a small measured effect from an edit with no computational path to the prediction?
5. **Bounded interpretation:** Do participants avoid assigning a general semantic role to a head from one prompt?

## Gate criteria

Set these before collecting data and do not change them afterward:

- At least five participants complete all three tasks and the concept probes. Run at least two sessions on a phone if possible.
- For questions 2–4 (probes C1–C3), at least four of five participants (or the same proportion of a larger group) give a **correct** answer after using the tool, without a facilitator hint. A **partial** answer, or a correct answer after a hint, does not count toward the gate; both are reported.
- Any misconception seen in two or more participants becomes an issue. A change to the interface or wording must be retested with new participants before the gate is considered met.
- Prediction accuracy (question 1) is reported but has no threshold. Wrong predictions that participants then explain correctly are a useful outcome.

## Participants and ethics

- Target: people who know what a neural network is and have heard of transformers or attention but have not studied mechanistic interpretability. Record only an experience band (for example "took an ML course", "uses LLMs, no ML course", "ML practitioner") and the device used.
- Obtain informed consent. Participation is voluntary, and participants may stop at any time. Record audio or screen only with explicit consent.
- Use participant IDs (P01, P02, …). Do not store names, emails, or identifying details in the repository. Quote participants only with consent and without identifying information.

## Materials

- The session page <https://jtmcbride.github.io/transformer-scope/study.html>. It is not linked from the application. It follows this script and records the deployed commit. Each prediction card is shown and its answers locked before the real application is loaded with that recipe, so the participant cannot see the result first. The page reads the result the participant sees, and exports one JSON record per participant. An unfinished session is kept on the device and can be resumed after a reload.
- Use one device per session and hand it to the participant for the tasks. The facilitator types think-aloud answers into the free-text fields. The page stores only the participant ID, experience band, device, consent, and answers. Do not type names or other identifying details into any field.
- Without the session page, use printed prediction cards (below) and cover the result panel until each prediction is recorded. Selecting a prepared experiment shows its result immediately.
- A copy of the findings template.

## Session script (about 35 minutes)

### 1. Introduction (3 minutes, read verbatim)

> "This tool runs a small trained transformer in your browser. It was trained on one task: read three key–value pairs, such as `A = 0 ; B = 3 ; C = 1`, then answer a query such as `? A =` with that key's value. Each layer contains several attention heads. Each head computes attention weights over earlier positions, combines information using those weights, and adds its output to the computation. You can change how strongly one head's output is added: 1 is normal, 0 removes it, and 2 doubles it. We are testing the tool, not you. Please think aloud."

Do not explain residual streams, causal masking, or where scaling happens. Answer interface questions ("where is the head selector?") but not conceptual ones until the debrief.

### 2. Orientation (4 minutes)

Have the participant run the input `A = 1 ; B = 2 ; C = 3 ; ? C =` with **Run input**, find the predicted answer, select a token, and move the strength slider once. Note any interface difficulty.

### 3. Tasks

For each task: show the prediction card, record the answers, then let the participant run the experiment (or select the prepared experiment) and look at the result. Ask the follow-up questions. The UI numbers layers and heads from 1; token positions start at 0, and position 0 is the beginning token `<BOS>`.

**Task 1: a substantial effect** (prepared experiment "Ablation changes the answer")

- Input `A = 0 ; B = 3 ; C = 0 ; ? B =`; prediction after position 15 (the final `=`); correct answer `3`.
- Intervention: Layer 1, Head 4, all positions, strength 0.

**Task 2: a reachable edit with little effect** (prepared experiment "A reachable edit has little effect")

- Input `A = 2 ; C = 0 ; B = 3 ; ? A =`; prediction after position 15; correct answer `2`.
- Intervention: Layer 1, Head 2, position 0 (`<BOS>`) only, strength 0.

**Task 3: no computational path** (prepared experiment "An earlier final-layer edit cannot reach the answer")

- Input `A = 0 ; B = 3 ; C = 0 ; ? B =`; prediction after position 15; correct answer `3`.
- Intervention: Layer 2 (the final layer), Head 1, position 0 (`<BOS>`) only, strength 0.

Prediction card questions (identical for each task):

- P1. Will the probability of the correct answer go **up**, **down**, or **stay the same**?
- P2. How much? **No change** / **tiny** (under 1 percentage point) / **moderate** / **large** (over 50 percentage points).
- P3. Why? (free text; prompt with "What would have to happen inside the model for the answer to change?")
- P4. Will this head's attention weights change? **Yes** / **no** / **unsure**, and why.

After the result:

- R1. Did the result match your prediction? What explains any difference?
- R2. (Tasks 2 and 3 only) The prediction barely changed or did not change. Does the tool say why?

### 4. Concept probes (8 minutes)

Ask these after all three tasks, before the debrief. Record verbatim answers. Participants may use the tool, including the attention matrix, cell inspector, and computation vectors.

- C1. **Attention invariance.** "When you set Head 4 in Layer 1 to 0 in Task 1, what happened to that head's attention weights? Why?" Ask the participant to check one attention cell at strength 1 and at strength 0.
- C2. **Propagation.** "In Task 1 the head's output changed. List what else in the model changed as a result, in order, up to the final prediction."
- C3. **Small effect versus no path.** "Tasks 2 and 3 both showed little or no change. Are they the same kind of result? Could either show a larger change for a different input?"
- C4. **Interpretation.** "Based on Task 1, what does Layer 1 Head 4 do?" Follow up: "How confident are you, and what would you need to check?"
- C5. **Transfer** (answer before running): "On the Task 1 input, could editing Layer 2, Head 1 **at position 15 only** change the prediction after position 15? What about **at position 14 only**?" Then let the participant check.

### 5. Debrief (3 minutes)

Explain the correct answers to any probes the participant missed and ask which part of the screen would have helped them. Record suggestions.

## Coding answers

Two people should code answers independently where possible, then resolve disagreements; record the number of disagreements. Use these definitions:

| Probe | Correct | Partial | Misconception (examples) |
| --- | --- | --- | --- |
| C1 | Weights are unchanged, because scaling acts on the head's output after attention weights have been applied (weights come from queries and keys). | Says unchanged without a reason, or with a vague reason. | "The head stops attending"; "weights become 0"; "the weights are redistributed to other tokens." |
| C2 | Head output → combined attention output/projection → residual stream → MLP → later layers (including later-layer attention at later positions) → final normalization/logits → probabilities. | Names the logits or "later layers" but not the residual/later-attention route. | "Only that head's attention map changes"; "it changes the weights of the model." |
| C3 | Task 2 has a path (through Layer 2 attention from position 0 to position 15) but a small measured effect on this input; a different input could show more. Task 3 cannot affect position 15 because the final layer has no later attention to move information between positions. | Recognizes a difference but cannot state it. | "Both heads are unimportant"; "Task 3 was not computed"; "both are the same." |
| C4 | The head is important for **this input's** answer under ablation; a general role would require many inputs and other tests. | Hedges but still names a specific role. | Assigns a definite role ("it is the key-lookup head") from one example. |
| C5 | Position 15: yes (same position, before the output). Position 14: no (final layer, earlier position). | One of the two is correct. | Both answered by "it's the final layer, so no" or "everything is connected, so yes". |

Prediction answers (P1/P2) are scored separately against the observed live result: direction (up/down/same) and magnitude band. P4 is correct only if the participant says no.

## Summarizing

1. Keep exported session files outside the repository. Verbatim answers can contain identifying details.
2. Two coders assign C/P/M/N to each probe in a codes file, for example `{ "P01": { "C1": "C", "C2": "P", "C3": "C", "C4": "M", "C5": "C" } }`, and resolve disagreements first.
3. Run `node scripts/summarize_study.mjs --codes codes.json sessions/*.json > tables.md`. The script validates every record, then produces the participant, prediction, probe, and gate tables for the findings template. Without codes it reports the gate as not evaluated.
4. Copy the tables into the findings document, remove anything identifying from the quotes, and commit only that document.

Prediction scoring is mechanical and uses the result shown at reveal. "Same" and "no change" mean an exactly zero distribution change (JSD 0). Otherwise the direction is the sign of the displayed Δ log p, which keeps the sign of a rounded value such as `-0.0000`. "Tiny" means a non-zero change under 1 percentage point, and "large" means over 50 points. Under these rules, Task 2 is observed as down/tiny and Task 3 as same/none.

## Reporting

- Report counts ("4 of 5"), not percentages, for fewer than 10 participants.
- Separate unprompted correct answers from answers given after a facilitator hint.
- Report every misconception and interface problem, including those seen once, with the participant ID and task.
- Do not generalize to a population, rank heads, or describe Layer 1 Head 4 as having a semantic function. The recipes are selected examples from one checkpoint, and each task uses one prompt.
- Record changes made in response to findings, and which participants saw which version (deployed commit).
