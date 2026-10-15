# TinyStories real-text deployment

This release ships the project's locally trained 48,000-step TinyStories checkpoint as an experimental language model. It is the existing four-layer pre-LN transformer trained on the first 250,000 TinyStories V2 stories, with a 2,048-token byte-level BPE tokenizer and a 64-token context including BOS. It uses the same inspectable float32 engine as the symbol models. It is distinct from the public `roneneldan/TinyStories-1M` GPT-Neo checkpoint.

The project owner's request to deploy real text authorizes this experimental release. The original acceptance gate stays failed, and the revised name probes remain a draft. No new training, threshold change, reranking, or revised name-probe evaluation is used to characterize this release as accepted. See `text-checkpoint-proposal.md` for both training runs and their failures.

## Exact checkpoint

- Model ID: `text-tinystories-seed-2026`.
- Revision: `23ee47f92cc78cc581abad5c53015276306d0f74ad04449527631c67b01666f6`.
- Weights SHA-256: `0011a5f8e314f63325a7231dfa60664e5bab37dbd8a27c1e5c03fab45996f294`.
- Weights: 4,254,720 bytes, 1,063,680 parameters.
- Tokenizer SHA-256: `bea77691157fbd7c2e339aa80e421057392fdda28dd003625336ed70dd0fc551`.
- Selected step: 48,000, according to the existing validation-loss rule.
- Mean validation loss: 1.772 nats/token; bigram 3.410; unigram 5.815.
- Original name-copying test: 112/200 (56%), below the unchanged 80% threshold. Pronoun agreement: 200/200. Full-run determinism and the untouched audit set remain unevaluated.

`public/models/tinystories/training.json` preserves the original run report, and `evaluation.json` preserves the release status and source provenance. `scripts/deploy_text.py` reads the existing export and independently packages fixtures without changing its weights. The data source is [TinyStories by Ronen Eldan and Yuanzhi Li](https://huggingface.co/datasets/roneneldan/TinyStories), pinned in `tinystories-source.json`; its dataset card identifies [CDLA Sharing 1.0](https://cdla.dev/sharing-1-0/). Training data and the optimizer checkpoint are not included in the static release. The model is trained locally; no OpenAI model weights or API are involved in inference.

## Text behavior

TinyStories is the default model. `?model=tinystories` selects it explicitly; `?model=retrieval` and `?model=verification` retain the symbol workspaces. The facilitated retrieval study requests its model explicitly.

Enter English text directly in Story prompt. The model computes probabilities for the next token, which can be a word, punctuation, or part of a word. The prompt is preserved exactly through UTF-8 encoding, export/import, source copying, and worker restart. No whitespace splitting, lowercasing, marker substitution, or silent truncation is applied. A passage exceeding 63 text tokens is rejected while retaining the last completed experiment. The display marks spaces with `␠`, newlines with `↵`, and incomplete UTF-8 token pieces with their byte values; these labels never become input text.

Example buttons fill a draft. Run input executes it and establishes the immutable baseline. The initial example starts at normal head strength; subsequent edits use the same head scaling, attention redirect, residual copy, inspection, tracing, and replay interfaces. The target defaults to the baseline's most probable next token for a new text input, while saved experiments preserve their explicit targets.

The visual flow shows one of the four layers at a time, with previous/next navigation and explicit labels for earlier and later layers. Navigation leaves the completed experiment intact; the Detailed graph still shows every layer.

The diagram predicts one next token. The continuation panel can now feed each chosen token back into the model, with step/play/stop, seeded sampling, and original/edited branches. See `controlled-continuation.md` for the sampling and stopping contract. There is no rolling context; stories remain bounded by this checkpoint's 64-token context.

## Verification and resource limits

Independent Python fixtures cover three text prompts, including punctuation, a newline, accented text, and an emoji. They include baseline logits, early and late head interventions, and a final-layer earlier-position no-path control. CPU engine checks compare both cached suffix and forced full execution against every reference logit under the existing `2e-5 + 1e-5 * abs(expected)` bound.

The on-demand PyTorch numerical probe checks captured forward stages, complete gradients, loss, and Adam updates under the existing bounds. It is about 47.3 MB and is loaded only when Verify against PyTorch is requested. The trained weights and tokenizer are much smaller; ordinary text entry does not fetch the probe. Measured CPU/WebGL reports are in `tinystories-numerical-results.json`. Zero retained tensor growth is checked after the worker cases and after the numerical probe. Maximum absolute stage error is reported alongside the scale-aware checks; an absolute maximum alone does not define the allowed error at every element.

Tokenizer parity retains the independent 163-case fixture and the all-validation-story parity runner. End-to-end browser coverage includes text edits, exact Unicode export, model switching, live replay, head interventions, restart preserving completed text rather than drafts, and actionable context-limit rejection. A physical mobile-device check and broader language-quality evaluation remain outstanding.

## Named-host timing and automatic backend

`tinystories-latency-results.json` records four sequential browser/backend benchmarks on the M1 Pro Mac, using the 17-token default prompt, early/middle/late layers (0, 2, 3), five warm-up edits and 40 measured edits per layer, plus alternating full/cached RPC samples. The reported frame opportunity adds two animation callbacks to the DOM-commit time; it does not measure physical display or touch latency.

Chromium cold readiness was 455 ms CPU and 2,097 ms WebGL; warm frame-opportunity p95s were at most 68 ms on either backend. WebKit cold readiness was 1,301 ms CPU and 476 ms WebGL; warm frame-opportunity p95 reached 317 ms on CPU and stayed at most 51 ms on WebGL. Automatic selection therefore prefers CPU for this exact revision on Chromium and requests WebGL elsewhere, with the existing CPU fallback when a float32 worker context is unavailable. Explicit `?backend=cpu` and `?backend=webgl` remain available. The choice is calibrated to this revision and these recorded desktop measurements; other devices and longer prompts require their own measurements.
