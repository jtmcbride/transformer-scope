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
