# Controlled associative retrieval

The shipped model is an educational symbol task, not a general language checkpoint.

## Task and split

An input is BOS followed by distinct key/value bindings and a query, for example:

```
A = 0 ; B = 3 ; C = 1 ; ? A =
```

The correct next token is `0`. Vocabulary order is BOS, A–D, 0–3, `=`, `;`, `?`. Values are independent, uniformly drawn from four symbols, with repeats allowed. Bindings use unique keys; the query selects any binding except the final one. This prevents copying the most recent value from solving the task. Training and ordinary evaluation use three bindings (16 input tokens including BOS).

`scripts/retrieval_data.py` exhaustively enumerates 3,072 such inputs. Split assignment hashes the salt `retrieval-v1-seed-2026`, a colon, and comma-separated token IDs with SHA256. The first eight digest bytes, interpreted big-endian, modulo 100 select train (<70), validation (<85), or test. Counts: 2,130 train, 489 validation, 453 test. Sequence digests are stored in `public/models/retrieval/evaluation.json`.

Dataset tests verify targets, unique keys, exact split disjointness, and that each identical seven-token local context has all four target answers equally represented in the full enumeration. The train-only local baseline predicts the majority target for the final seven tokens (last binding plus query), choosing value 0 for unseen contexts. It scores 15.23% on the finite held-out split; this is below chance because majority imbalances in a sampled training partition can reverse in its complementary test partition. The theoretical uniform-answer reference remains 25% accuracy and ln(4) loss.

Test results were used as an export quality gate during development, including rejecting the first candidate. They establish held-out-sequence performance, not an untouched final audit or broad language capability.

## Training evidence

The initial candidate (seed 2026; 2 layers, 2 heads, dimension 32, MLP 128; Adam learning rate .003; 2,000 steps) plateaued near 61% validation accuracy and scored 60.71% on held-out inputs. It was not exported. This motivated greater capacity and a lower learning rate; the experiment does not isolate the effect of either change.

The accepted candidate uses 2 layers, 4 heads, dimension 64, head dimension 16, MLP 256, context 32, and 102,912 parameters (411,648 float32 weight bytes). Training uses PyTorch 2.9.1, seed 2026, two CPU threads, deterministic algorithms, token/position initialization std .1, batch size 64 sampled with replacement, and Adam at .001, betas .9/.999, epsilon 1e-5, gradient norm clipping 1. Cross entropy over all 12 symbols is supervised only at the final query position. Validation loss selects the best checkpoint every 100 steps; the selected step is 3,000. The full validation history is exported.

Ordinary held-out result: 453/453 correct, mean cross entropy about 0.0000982 nats, mean target probability 99.9902%. Validation accuracy is also 100%. Two-binding evaluation (192 inputs) scores 79.6875%; four-binding evaluation (18,432 inputs) scores 62.9232%. These lengths were absent from training and are reported separately. There are no unseen key or value symbols in these evaluations.

A second run using the pinned environment reproduced identical weights, tokenizer, and manifest SHA256 hashes. This is evidence for reproducibility on this host, not a guarantee of identical floating-point optimization on every platform. The selected local snapshot contains weights/config/seed/step, not resumable Adam/RNG/sampler state.

## Prepared experiments

Recipes are discovered on the first 96 validation inputs in generator order, scanning both layers and all four heads. The large-effect recipe selects an all-position ablation with a confident baseline and the largest reduction in correct-answer log probability. The small-effect recipe selects a first-layer position-0 ablation with a confident baseline and the smallest JSD. The third recipe edits position 0 in the final layer on the large-effect input, making the later answer structurally unreachable from that edit.

Each recipe records its exact text/token IDs, zero-based layer/head/positions, target, factor, checkpoint revision, baseline/modified logits and probabilities, Δ log probability, JSD, and max logit change. The browser recomputes these quantities; stored measurements are separately labeled. Numerical checks use the original tolerances. The substantial effect is many orders of magnitude above observed reference error. The tiny control effect is not interpreted as a reliably resolved causal effect.

These recipes intentionally select examples and extremes. They do not estimate average head importance, prove semantic specialization, or establish that heads act independently. The no-path control is an architecture invariant; the reachable control illustrates why a path need not have a large measured effect.

## Head-ablation coverage

The three recipes are selected examples. To show how representative they are, `npm run analysis:heads` ablates one head at a time (factor 0) on all 453 held-out inputs and records the change at the final query position. It uses the verified engine on CPU. The TypeScript split generator reproduces the recorded split counts and SHA-256 digests exactly (engine test `split.test.ts`). Raw results: `docs/head-coverage-results.json` (Node 22.22, TF.js 4.22.0 CPU, Linux x86_64; recorded 2026-10-12).

| Ablated head (all positions) | Answer changed (of 453) | Median Δ log p | 10th percentile | Most negative | Mean correct-answer probability after |
| --- | --- | --- | --- | --- | --- |
| Layer 1, Head 1 | 33 | -6.4e-06 | -0.155 | -10.2 | 92.1% |
| Layer 1, Head 2 | 57 | -1.7e-05 | -1.63 | -7.7 | 87.8% |
| Layer 1, Head 3 | 12 | -1.2e-05 | -0.00419 | -4.45 | 97.0% |
| Layer 1, Head 4 | 218 | -0.28 | -10.7 | -12 | 52.0% |
| Layer 2, Head 1 | 24 | -1.4e-05 | -0.0119 | -5.51 | 94.9% |
| Layer 2, Head 2 | 209 | -0.4 | -10.2 | -13 | 52.7% |
| Layer 2, Head 3 | 0 | -7.4e-06 | -5.16e-05 | -0.000333 | 100.0% |
| Layer 2, Head 4 | 0 | -4.3e-06 | -0.000205 | -0.00293 | 100.0% |

- Ablating Layer 1 Head 4 changes the answer on fewer than half of the inputs, and its median effect is small. The large-effect recipe (−11.78 nats) lies in the extreme tail of this distribution. It is not a typical outcome.
- Layer 2 Head 2 is comparably decisive. Counting, for each input, which single all-position ablation lowers the correct answer most: Layer 2 Head 2 195, Layer 1 Head 4 192, Layer 1 Head 2 29, Layer 2 Head 1 28, Layer 1 Head 3 6, Layer 1 Head 1 3.
- Ablating any layer-1 head at position 0 only never changed an answer. The largest change was below 6e-6 nats, which is below the established 2e-5 logit tolerance and not interpreted. The small-effect recipe is typical in this respect.
- Ablating any final-layer head at position 0 left the prediction exactly unchanged on all 453 inputs (JSD 0, maximum logit change 0), as the architecture requires.
- Changes smaller than about 1e-5 nats, including the small positive values, are within numerical error and are not interpreted.

These measurements cover one checkpoint, one task, one prediction position, factor 0, and single heads; they do not measure interactions. They show that a head's measured importance varies widely across inputs. They do not establish a semantic role for any head. Intervention results here use the engine whose forward, cached, and intervention paths match PyTorch on the recorded fixtures and recipes; these 7,248 comparisons were not separately recomputed in PyTorch.

## Reproduce and inspect

```
uv sync --locked
npm run test:data
npm run train:retrieval
npm test
npm run build
npm run test:browser -- --project=chromium --project=webkit
```

`scripts/export_artifact.py` exports float32 input-output matrices and a probe against a separate model clone. Probe loss uses the existing mean full-sequence loss contract to verify all gradients/Adam; this differs deliberately from final-answer-only offline training. Tests never retrain or silently regenerate expected values.

The app defaults to the trained model and keeps the untrained verification model selectable. `?model=verification` opens the original checkpoint directly. All predictions and interventions run in the browser worker. Static probe arrays are fetched only when verification is requested; ordinary startup loads about 402 KiB of weights plus small metadata files. The attention matrix scrolls within its panel on narrow screens.
