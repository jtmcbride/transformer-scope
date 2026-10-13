"""Gating probes for the text checkpoint (docs/text-checkpoint-proposal.md). Prompts come from our own templates.

Usage: .venv/bin/python scripts/text_probes.py [--model .cache/text/export]
Writes probes.json next to the model. Thresholds are fixed here and must not be lowered after a run.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from reference_model import Config, ReferenceTransformer
from text_tokenizer import Tokenizer

THRESHOLD = .8
PROMPTS = 200
SEED = 2027
# Names that are one token with a leading space in the trained tokenizer (checked at run time).
GIRLS = ['Lily', 'Sue', 'Anna', 'Mia', 'Lucy', 'Sara', 'Emma', 'Amy', 'Jane', 'Sally', 'Lila']
BOYS = ['Tom', 'Ben', 'Max', 'Tim', 'Jack', 'Bob', 'Joe', 'Billy', 'Timmy', 'Tommy']
PLACES = ['park', 'store', 'beach', 'forest', 'garden', 'zoo', 'farm', 'lake']
OBJECTS = ['ball', 'flower', 'book', 'toy', 'cookie', 'hat', 'apple', 'kite']
COPYING = [
    '{A} and {B} went to the {place}. {B} gave a {object} to',
    'Once upon a time, {A} and {B} found a {object} at the {place}. {B} was happy and said thank you to',
    'One day, {A} met {B} at the {place}. {B} wanted to share a {object} with',
]
PRONOUN = 'Once upon a time, there was a little {kind} named {name}. One day,'


def load_model(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    tokenizer = Tokenizer(json.loads((directory / 'tokenizer.json').read_text())['merges'])
    model = ReferenceTransformer(Config(**{key: manifest['config'][key] for key in Config.__dataclass_fields__}))
    payload = (directory / 'weights.bin').read_bytes()
    parameters = model.exported_parameters()
    with torch.no_grad():
        for entry in manifest['tensors']:
            values = np.frombuffer(payload, dtype='<f4', count=entry['byteLength'] // 4, offset=entry['byteOffset']).reshape(entry['shape'])
            parameter, transpose = parameters[entry['name']]
            parameter.copy_(torch.from_numpy(values.T.copy() if transpose else values.copy()))
    return model.eval(), tokenizer, manifest


def single(tokenizer, word):
    ids = tokenizer.encode(' ' + word, bos=False)
    if len(ids) != 1: raise SystemExit(f'" {word}" is {len(ids)} tokens; probes need single-token words.')
    return ids[0]


@torch.no_grad()
def next_log_probs(model, tokenizer, text):
    ids = tokenizer.encode(text)
    if len(ids) > model.config.contextLength: raise SystemExit(f'Prompt exceeds the context: {text}')
    logits, _ = model(torch.tensor([ids]), capture_activations=False)
    return logits[0, -1].double().log_softmax(-1)


def name_copying(model, tokenizer, generator):
    names = GIRLS + BOYS
    results = []
    for i in range(PROMPTS):
        a, b = generator.sample(names, 2)
        absent = generator.choice([n for n in names if n not in (a, b)])
        prompt = COPYING[i % len(COPYING)].format(A=a, B=b, place=generator.choice(PLACES), object=generator.choice(OBJECTS))
        scores = next_log_probs(model, tokenizer, prompt)
        top = int(scores.argmax())
        results.append(dict(prompt=prompt, expected=a, top=tokenizer.display(top), correct=top == single(tokenizer, a),
                            pExpected=float(scores[single(tokenizer, a)].exp()), pOther=float(scores[single(tokenizer, b)].exp()),
                            pAbsentName=float(scores[single(tokenizer, absent)].exp())))
    rate = sum(r['correct'] for r in results) / len(results)
    return dict(description='The earlier-mentioned name is the top-1 next token; control: probability of a name that never appeared.',
                templates=COPYING, prompts=len(results), top1Rate=rate, threshold=THRESHOLD, passed=rate >= THRESHOLD,
                meanExpectedProbability=float(np.mean([r['pExpected'] for r in results])),
                meanAbsentNameProbability=float(np.mean([r['pAbsentName'] for r in results])), results=results)


def pronoun_agreement(model, tokenizer, generator):
    he, she = single(tokenizer, 'he'), single(tokenizer, 'she')
    results = []
    for i in range(PROMPTS):
        kind = 'boy' if i % 2 == 0 else 'girl'
        name = generator.choice(BOYS if kind == 'boy' else GIRLS)
        prompt = PRONOUN.format(kind=kind, name=name)
        scores = next_log_probs(model, tokenizer, prompt)
        expected, other = (he, she) if kind == 'boy' else (she, he)
        results.append(dict(prompt=prompt, kind=kind, correct=bool(scores[expected] > scores[other]),
                            pHe=float(scores[he].exp()), pShe=float(scores[she].exp())))
    rate = sum(r['correct'] for r in results) / len(results)
    return dict(description='p(" he") > p(" she") after a named boy, and the reverse after a named girl.', template=PRONOUN,
                prompts=len(results), agreementRate=rate, threshold=THRESHOLD, passed=rate >= THRESHOLD, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(Path(__file__).resolve().parents[1] / '.cache/text/export'))
    directory = Path(parser.parse_args().model)
    model, tokenizer, manifest = load_model(directory)
    report = dict(modelRevision=manifest['modelRevision'], seed=SEED,
                  nameCopying=name_copying(model, tokenizer, random.Random(SEED)),
                  pronounAgreement=pronoun_agreement(model, tokenizer, random.Random(SEED + 1)))
    (directory / 'probes.json').write_text(json.dumps(report, indent=2) + '\n')
    for key in ('nameCopying', 'pronounAgreement'):
        probe = report[key]
        rate = probe.get('top1Rate', probe.get('agreementRate'))
        print(f"{key}: {rate:.1%} (threshold {THRESHOLD:.0%}) — {'passed' if probe['passed'] else 'NOT passed; report as a finding'}")


if __name__ == '__main__':
    main()
