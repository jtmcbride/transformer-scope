"""Train the first text checkpoint on TinyStories V2 (see docs/text-checkpoint-proposal.md).

Prerequisites: scripts/text_data.py, then scripts/text_tokenizer.py. Example (defaults follow the approved proposal):
  .venv/bin/python scripts/train_text.py                 # about 24,000 steps of 32 x 64 tokens
  .venv/bin/python scripts/train_text.py --resume        # continue from .cache/text/checkpoint.pt
  .venv/bin/python scripts/train_text.py --steps 200 --eval-every 100 --output .cache/text/smoke   # quick check
Writes the selected weights, tokenizer, manifest, and training record to --output (default .cache/text/export).
Copying them into public/models/text is a separate, reviewed step after the probes pass.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from reference_model import Config, ReferenceTransformer
from export_artifact import write_artifact
from text_data import OUTPUT as DATA, load as load_stories
from text_tokenizer import Tokenizer, load as load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
BOS, EOS, PAD = 0, 1, 2


def tokenize_corpus(tokenizer, name):
    """Concatenated story tokens and offsets, cached by tokenizer and subset hash."""
    stories = load_stories(name)
    key = hashlib.sha256(tokenizer.serialized() + json.dumps([len(stories), stories[0], stories[-1]]).encode()).hexdigest()[:16]
    path = DATA / f'{name}-tokens-{key}.npz'
    if path.exists():
        cached = np.load(path)
        return cached['tokens'], cached['offsets']
    encoded = [tokenizer.encode(story, bos=False) for story in stories]
    offsets = np.cumsum([0] + [len(ids) for ids in encoded]).astype(np.int64)
    tokens = np.fromiter((i for ids in encoded for i in ids), dtype=np.uint16, count=int(offsets[-1]))
    np.savez(path, tokens=tokens, offsets=offsets)
    return tokens, offsets


def window(tokens, offsets, story, start, length):
    """[BOS] + story tokens from start, then EOS, padded with PAD to length + 1 (inputs and shifted targets)."""
    body = tokens[offsets[story] + start: offsets[story + 1]].tolist()
    sequence = ([BOS] + body + [EOS])[:length + 1]
    return sequence + [PAD] * (length + 1 - len(sequence))


def sample(tokens, offsets, generator, batch, length):
    stories = len(offsets) - 1
    rows = []
    for _ in range(batch):
        story = int(torch.randint(stories, (1,), generator=generator))
        size = int(offsets[story + 1] - offsets[story])
        # Half the windows begin at the story's start, matching how the screen is used.
        start = 0 if size <= 1 or torch.rand(1, generator=generator).item() < .5 else int(torch.randint(size, (1,), generator=generator))
        rows.append(window(tokens, offsets, story, start, length))
    data = torch.tensor(rows)
    return data[:, :-1], data[:, 1:]


def evaluation_windows(tokens, offsets, length, count, seed):
    """Fixed validation windows: story openings, then seeded random offsets."""
    generator = torch.Generator().manual_seed(seed)
    stories = min(count, len(offsets) - 1)
    openings = [window(tokens, offsets, s, 0, length) for s in range(stories)]
    random_rows = []
    for s in range(stories):
        size = int(offsets[s + 1] - offsets[s])
        random_rows.append(window(tokens, offsets, s, int(torch.randint(max(size, 1), (1,), generator=generator)), length))
    return {name: torch.tensor(rows) for name, rows in (('openings', openings), ('randomOffsets', random_rows))}


@torch.no_grad()
def evaluate(model, windows):
    model.eval()
    results = {}
    for name, data in windows.items():
        total, count = 0., 0
        for batch in data.split(128):
            logits, _ = model(batch[:, :-1], capture_activations=False)
            targets = batch[:, 1:]
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD, reduction='sum')
            total += losses.item(); count += int((targets != PAD).sum())
        results[name] = total / count
    results['mean'] = (results['openings'] + results['randomOffsets']) / 2
    model.train()
    return results


def ngram_baselines(tokens, offsets, vocabulary, windows, smoothing=.1):
    """Unigram and add-k bigram models from the training stream (BOS→first token and last→EOS included), scored on the same windows."""
    previous = np.concatenate([[BOS], tokens])
    stream_prev, stream_next = [], []
    for s in range(len(offsets) - 1):
        body = tokens[offsets[s]:offsets[s + 1]]
        stream_prev.append(np.concatenate([[BOS], body])); stream_next.append(np.concatenate([body, [EOS]]))
    prev, nxt = np.concatenate(stream_prev).astype(np.int64), np.concatenate(stream_next).astype(np.int64)
    unigram = np.bincount(nxt, minlength=vocabulary).astype(np.float64) + smoothing
    unigram /= unigram.sum()
    bigram = np.zeros((vocabulary, vocabulary)); np.add.at(bigram, (prev, nxt), 1); bigram += smoothing
    bigram /= bigram.sum(1, keepdims=True)
    results = {}
    for name, data in windows.items():
        x, y = data[:, :-1].numpy().reshape(-1), data[:, 1:].numpy().reshape(-1)
        keep = y != PAD
        results[name] = dict(unigram=float(-np.log(unigram[y[keep]]).mean()), bigram=float(-np.log(bigram[x[keep], y[keep]]).mean()))
    for kind in ('unigram', 'bigram'): results.setdefault('mean', {})[kind] = (results['openings'][kind] + results['randomOffsets'][kind]) / 2
    return dict(smoothing=smoothing, nats=results)


def learning_rate(step, steps, peak, warmup, floor):
    if step < warmup: return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(1., progress))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add('--steps', type=int, default=24_000); add('--batch', type=int, default=32); add('--context', type=int, default=64)
    add('--layers', type=int, default=4); add('--heads', type=int, default=4); add('--model-dim', type=int, default=128); add('--mlp-dim', type=int, default=512)
    add('--embedding-std', type=float, default=.02); add('--learning-rate', type=float, default=2e-3); add('--warmup', type=int, default=200); add('--floor', type=float, default=.1)
    add('--eval-every', type=int, default=500); add('--eval-stories', type=int, default=2000)
    add('--seed', type=int, default=2026); add('--threads', type=int, default=4); add('--device', default='cpu')
    add('--resume', action='store_true'); add('--stop-after', type=int, default=None, help='Save a checkpoint and exit at this step (resume later).'); add('--output', default=str(DATA / 'export'))
    args = parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    tokenizer = load_tokenizer(DATA / 'bpe.json')
    vocabulary = 3 + 256 + len(tokenizer.merges)
    train_tokens, train_offsets = tokenize_corpus(tokenizer, 'train')
    valid_tokens, valid_offsets = tokenize_corpus(tokenizer, 'validation')
    windows = evaluation_windows(valid_tokens, valid_offsets, args.context, args.eval_stories, args.seed)
    config = Config(layers=args.layers, heads=args.heads, modelDim=args.model_dim, mlpDim=args.mlp_dim, vocabularySize=vocabulary, contextLength=args.context)
    model = ReferenceTransformer(config)
    # Default N(0, 1) tied embeddings give initial logits far from uniform (about 65 nats/token at d=128).
    torch.nn.init.normal_(model.token.weight, std=args.embedding_std); torch.nn.init.normal_(model.position.weight, std=args.embedding_std)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.9, .999), eps=1e-8)
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint_path, step, history, best = DATA / 'checkpoint.pt', 0, [], None
    if args.resume:
        saved = torch.load(checkpoint_path, weights_only=False)
        if saved['arguments'] != {k: v for k, v in vars(args).items() if k not in ('resume', 'stop_after')}: raise SystemExit('Resume requires the same arguments.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); generator.set_state(saved['generator'])
        step, history, best = saved['step'], saved['history'], saved['best']
        print(f'Resumed at step {step}.')
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters; {len(train_offsets) - 1:,} training stories, {len(train_tokens):,} tokens.')
    started = time.perf_counter()
    model.train()
    while step < args.steps:
        for group in optimizer.param_groups: group['lr'] = learning_rate(step, args.steps, args.learning_rate, args.warmup, args.floor)
        x, y = sample(train_tokens, train_offsets, generator, args.batch, args.context)
        logits, _ = model(x.to(args.device), capture_activations=False)
        loss = F.cross_entropy(logits.reshape(-1, vocabulary), y.to(args.device).reshape(-1), ignore_index=PAD)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            losses = evaluate(model, windows)
            history.append(dict(step=step, trainingLoss=loss.item(), validation=losses, seconds=time.perf_counter() - started))
            if best is None or losses['mean'] < best['validation']['mean']:
                best = dict(step=step, validation=losses, model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            torch.save(dict(arguments={k: v for k, v in vars(args).items() if k not in ('resume', 'stop_after')}, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            generator=generator.get_state(), step=step, history=history, best=best), checkpoint_path)
            print(f"step {step}: train {loss.item():.3f}, validation openings {losses['openings']:.3f} / random {losses['randomOffsets']:.3f} nats/token")
            if args.stop_after is not None and step >= args.stop_after:
                print(f'Stopped after step {step}; continue with --resume.'); return
    model.load_state_dict(best['model']); model.cpu()
    baselines = ngram_baselines(train_tokens, train_offsets, vocabulary, windows)
    prepared = json.loads((DATA / 'prepared.json').read_text())
    output = Path(args.output)
    provenance = dict(purpose='TinyStories V2 text checkpoint', seed=args.seed, torchVersion=torch.__version__, generatorVersion=1,
                      data=prepared['source'], subsets=prepared['subsets'], selectedStep=best['step'])
    manifest = write_artifact(model, output, None, f'text-tinystories-seed-{args.seed}', provenance, tokenizer.serialized())
    record = dict(modelRevision=manifest['modelRevision'], arguments={k: v for k, v in vars(args).items() if k not in ('resume', 'stop_after')},
                  environment=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(), processor=platform.processor() or platform.machine(),
                                   cpuCount=os.cpu_count(), threads=args.threads, device=args.device),
                  tokens=dict(train=int(len(train_tokens)), validation=int(len(valid_tokens)), seenDuringTraining=args.steps * args.batch * args.context),
                  selected=dict(step=best['step'], validation=best['validation']), baselines=baselines,
                  criteria=dict(beatsBigramByOneNat=best['validation']['mean'] <= baselines['nats']['mean']['bigram'] - 1,
                                beatsUnigram=best['validation']['mean'] < baselines['nats']['mean']['unigram']),
                  history=history, weightsSha256=manifest['weightsSha256'])
    (output / 'training.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(selected=record['selected'], baselines=baselines['nats']['mean'], criteria=record['criteria'], weightsSha256=manifest['weightsSha256']), indent=2))


if __name__ == '__main__':
    main()
