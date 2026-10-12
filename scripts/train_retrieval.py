"""Offline CPU training, isolated held-out evaluation, and measured recipe discovery."""
import argparse
import copy
import json
import time
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from reference_model import Config, ReferenceTransformer
from retrieval_data import SYMBOLS, SPLIT_SALT, examples, partition, dataset_digest
from export_artifact import write_artifact, write_probe

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def tensor_data(items):
    return torch.tensor([item.tokens for item in items]), torch.tensor([item.target for item in items])


@torch.no_grad()
def evaluate(model, items):
    tokens, targets = tensor_data(items)
    logits = torch.cat([model(batch, capture_activations=False)[0][:, -1] for batch in tokens.split(128)])
    predictions = logits.argmax(-1)
    return dict(count=len(items), accuracy=(predictions == targets).float().mean().item(),
                lossNats=F.cross_entropy(logits, targets).item(),
                meanTargetProbability=logits.softmax(-1).gather(1, targets[:, None]).mean().item())


def local_predictor(training):
    # Last binding plus '? key =': cannot see the queried binding's independent value.
    counts = defaultdict(Counter)
    for item in training:
        counts[item.tokens[-7:]][item.target] += 1
    return {context: sorted(count, key=lambda target: (-count[target], target))[0] for context, count in counts.items()}


def local_accuracy(predictor, items):
    return sum(predictor.get(item.tokens[-7:], 5) == item.target for item in items) / len(items)


def intervention(layer, head, positions, factor=0):
    return dict(id='head-strength', kind='head.scale', layer=layer, head=head, factor=factor,
                tokens=dict(kind='all') if positions is None else dict(kind='indices', indices=positions))


@torch.no_grad()
def measure(model, item, patch, position=None):
    position = len(item.tokens) - 1 if position is None else position
    baseline = model(torch.tensor([item.tokens]), capture_activations=False)[0][0, position]
    modified = model(torch.tensor([item.tokens]), [patch], capture_activations=False)[0][0, position]
    # Float64 metrics match the browser distribution comparison contract.
    a, b = baseline.double().log_softmax(-1), modified.double().log_softmax(-1)
    pa, pb = a.exp(), b.exp()
    mixture = ((pa + pb) / 2).log()
    jsd = .5 * ((pa * (a - mixture)).sum() + (pb * (b - mixture)).sum())
    return dict(baselineLogits=baseline.tolist(), modifiedLogits=modified.tolist(),
                baselineTargetProbability=pa[item.target].item(), modifiedTargetProbability=pb[item.target].item(),
                baselineTargetLogProbability=a[item.target].item(), modifiedTargetLogProbability=b[item.target].item(),
                targetLogProbabilityDelta=(b[item.target] - a[item.target]).item(),
                jsDivergence=0. if torch.equal(baseline, modified) else max(0., jsd.item()), maxAbsoluteLogitChange=(modified - baseline).abs().max().item())


def recipes(model, manifest):
    # Discovery is restricted to validation; the test accuracy never selects a recipe.
    candidates = [item for item in examples() if partition(item) == 'validation'][:96]
    effect, small = None, None
    for item in candidates:
        for layer in range(model.config.layers):
            for head in range(model.config.heads):
                patch = intervention(layer, head, None)
                metrics = measure(model, item, patch)
                candidate = (item, patch, metrics)
                if metrics['baselineTargetProbability'] > .9 and (effect is None or metrics['targetLogProbabilityDelta'] < effect[2]['targetLogProbabilityDelta']):
                    effect = candidate
                # Earlier first-layer position can influence the final answer through layer 2.
                patch = intervention(0, head, [0])
                metrics = measure(model, item, patch)
                if metrics['baselineTargetProbability'] > .9 and (small is None or metrics['jsDivergence'] < small[2]['jsDivergence']):
                    small = (item, patch, metrics)
    if effect is None or small is None or effect[2]['targetLogProbabilityDelta'] > -.1:
        raise RuntimeError('Checkpoint did not produce a sufficiently strong measured intervention recipe.')
    if small[2]['jsDivergence'] > .001:
        raise RuntimeError('No reachable small-effect control found.')
    item = effect[0]
    no_path_patch = intervention(model.config.layers - 1, 0, [0])
    unreachable = (item, no_path_patch, measure(model, item, no_path_patch))
    if unreachable[2]['maxAbsoluteLogitChange'] != 0:
        raise RuntimeError('Unreachable recipe unexpectedly changed the final prediction.')
    result = []
    descriptions = [
        ('effect', 'Ablation changes the answer', 'This ablation reduces the correct answer probability on this example. It does not establish a general semantic role for the head.'),
        ('small', 'A reachable edit has little effect', 'An earlier first-layer edit has a path through the next attention layer, but this example shows little distribution change. Reachability does not guarantee a large effect.'),
        ('unreachable', 'An earlier final-layer edit cannot reach the answer', 'The final layer has no subsequent attention mixing. Editing its head output at position 0 cannot change the later query prediction.')]
    for (kind, title, interpretation), (item, patch, metrics) in zip(descriptions, [effect, small, unreachable]):
        result.append(dict(id=kind, title=title, interpretation=interpretation, modelRevision=manifest['modelRevision'],
                           discoverySplit='validation', text=item.text, tokenIds=list(item.tokens),
                           predictionPosition=len(item.tokens)-1, targetTokenId=item.target, intervention=patch, metrics=metrics))
    return dict(schemaVersion=1, modelRevision=manifest['modelRevision'], recipes=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=.001)
    args = parser.parse_args()
    if args.steps < 1 or args.layers < 2 or args.dim < 1 or args.heads < 1 or args.dim % args.heads or args.learning_rate <= 0:
        parser.error('Use positive steps/dim/heads/rate, at least two layers, and a dimension divisible by the heads.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    split = {name: [item for item in examples() if partition(item) == name] for name in ('train', 'validation', 'test')}
    model = ReferenceTransformer(Config(layers=args.layers, heads=args.heads, modelDim=args.dim, mlpDim=4*args.dim, vocabularySize=len(SYMBOLS), contextLength=32))
    torch.nn.init.normal_(model.token.weight, std=.1)
    torch.nn.init.normal_(model.position.weight, std=.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.9, .999), eps=1e-5)
    x, y = tensor_data(split['train'])
    history, best, best_loss, best_step = [], None, float('inf'), 0
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(x), (64,))
        logits, _ = model(x[indices], capture_activations=False)
        loss = F.cross_entropy(logits[:, -1], y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            score = evaluate(model, split['validation'])
            history.append(dict(step=step, trainingBatchLoss=loss.item(), validation=score))
            if score['lossNats'] < best_loss:
                best, best_loss, best_step = copy.deepcopy(model.state_dict()), score['lossNats'], step
            print(f"step {step}: validation accuracy {score['accuracy']:.3%}, loss {score['lossNats']:.4f}", flush=True)
    model.load_state_dict(best)
    test = evaluate(model, split['test'])
    attempt_path = ROOT / '.cache/retrieval/attempts.json'
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else []
    attempts.append(dict(settings=vars(args), selectedStep=best_step, validationLoss=best_loss, heldOut=test, history=history))
    attempt_path.write_text(json.dumps(attempts, indent=2) + '\n')
    if test['accuracy'] < .95:
        raise RuntimeError(f"Held-out accuracy gate failed: {test['accuracy']:.3%}; candidate is not exported.")
    output = ROOT / '.cache/retrieval/export'
    manifest = write_artifact(model, output, SYMBOLS, f'retrieval-seed-{args.seed}',
                              dict(purpose='trained controlled associative retrieval', seed=args.seed, torchVersion=torch.__version__,
                                   generatorVersion=1, selectedStep=best_step, objective='final-answer cross entropy', splitSalt=SPLIT_SALT))
    prepared = recipes(model, manifest)
    predictor = local_predictor(split['train'])
    evaluation = dict(schemaVersion=1, modelRevision=manifest['modelRevision'], seed=args.seed,
                      training=dict(steps=args.steps, selectedStep=best_step, batchSize=64, learningRate=args.learning_rate, beta1=.9, beta2=.999,
                                    epsilon=1e-5, gradientClipNorm=1., threads=2, embeddingInitializationStd=.1, deterministicAlgorithms=True, elapsedSeconds=time.monotonic()-start,
                                    objective='Cross entropy over all vocabulary tokens, supervised at final query position only.'),
                      split=dict(salt=SPLIT_SALT, rule='SHA256(salt + colon + comma-separated input token IDs), first 8 bytes big-endian modulo 100: train <70; validation <85; otherwise test.',
                                 digests={name: dataset_digest(items) for name, items in split.items()}, counts={name: len(items) for name, items in split.items()}),
                      heldOut=test, validation=evaluate(model, split['validation']),
                      chance=dict(accuracy=.25, lossNats=1.3862943611198906, description='Uniform over the four answer values; probability zero on grammar symbols.'),
                      localContext=dict(accuracy=local_accuracy(predictor, split['test']), description='Train-set majority lookup of final seven tokens (last binding and query); unseen contexts choose value 0. Queried binding is never last; its value is independent.'),
                      lengthGeneralization={str(n): evaluate(model, list(examples(n))) for n in (2, 4)}, history=history,
                      limitations=['Exhaustive finite symbol task; held-out sequences, not unseen keys or values.',
                                   'Training uses exactly three bindings. Length tests use two and four bindings and are reported separately.',
                                   'Recipe discovery uses up to 96 validation inputs and selects extremes; recipes are illustrations, not population effect estimates.'])
    (output / 'evaluation.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    (output / 'recipes.json').write_text(json.dumps(prepared, indent=2) + '\n')
    write_probe(model, output, prepared['recipes'][0]['tokenIds'], [prepared['recipes'][0]['intervention']])
    with torch.no_grad():
        tokens, _ = tensor_data(split['test'])
        held_out_logits = model(tokens, capture_activations=False)[0][:, -1]
    fixture = dict(modelRevision=manifest['modelRevision'], cases=[dict(tokenIds=list(item.tokens), target=item.target, logits=row.tolist())
                   for item, row in zip(split['test'], held_out_logits)])
    (ROOT / 'tests/fixtures/retrieval-evaluation.json').write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
    destination = ROOT / 'public/models/retrieval'
    destination.mkdir(parents=True, exist_ok=True)
    for file in output.iterdir():
        shutil.copyfile(file, destination / file.name)
    # Selected weights stay local; no optimizer resume state is claimed.
    local = ROOT / '.cache/retrieval'
    local.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=manifest['config'], seed=args.seed, selectedStep=best_step, state_dict=best), local / 'checkpoint.pt')
    print(json.dumps(dict(heldOut=test, lengthGeneralization=evaluation['lengthGeneralization'], modelRevision=manifest['modelRevision'],
                          recipes=[dict(id=r['id'], delta=r['metrics']['targetLogProbabilityDelta'], jsd=r['metrics']['jsDivergence']) for r in prepared['recipes']]), indent=2), flush=True)


if __name__ == '__main__':
    main()
