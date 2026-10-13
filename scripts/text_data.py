"""TinyStories V2 preparation: fetch pinned files, split stories, and fix the training, tokenizer, and audit subsets.

Usage: .venv/bin/python scripts/text_data.py [--train-stories N] [--audit-stories N]
Outputs go to .cache/text/ (not committed); prepared.json records counts and hashes.
"""
import argparse
import hashlib
import json
import urllib.request
from collections import deque
from pathlib import Path

root = Path(__file__).resolve().parents[1]
SOURCE = json.loads((root / 'docs/tinystories-source.json').read_text())
DATA = root / '.cache/tinystories'
OUTPUT = root / '.cache/text'
SEPARATOR = '<|endoftext|>'
DEFAULTS = dict(train_stories=250_000, audit_stories=2_000, tokenizer_stories=20_000)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(directory=DATA):
    """Download any missing pinned file and verify size and SHA-256 of every file."""
    directory.mkdir(parents=True, exist_ok=True)
    for entry in SOURCE['files']:
        path = directory / entry['name']
        if not path.exists():
            url = f"https://huggingface.co/datasets/{SOURCE['dataset']}/resolve/{SOURCE['revision']}/{entry['name']}"
            print(f'Downloading {url}')
            urllib.request.urlretrieve(url, path)
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise SystemExit(f'{path} does not match docs/tinystories-source.json; delete it and fetch again.')
    return directory


def normalize(story):
    return story.replace('\r\n', '\n').strip()


def stories(path, drop_leading_fragment, block_size=1 << 22):
    """Yield normalized, non-empty stories split on the separator, streaming the file."""
    buffer, first = '', True
    with path.open('r', encoding='utf-8', newline='') as handle:
        for block in iter(lambda: handle.read(block_size), ''):
            buffer += block
            *complete, buffer = buffer.split(SEPARATOR)
            for chunk in complete:
                if not (first and drop_leading_fragment) and normalize(chunk):
                    yield normalize(chunk)
                first = False
    if normalize(buffer) and not (first and drop_leading_fragment):
        yield normalize(buffer)


def digest(items):
    return hashlib.sha256('\n\x00\n'.join(items).encode()).hexdigest()


def prepare(train_stories, audit_stories, tokenizer_stories, directory=DATA, output=OUTPUT):
    """Training subset: the first train_stories stories. Audit: the final audit_stories stories, never trained on.
    Tokenizer data: the first tokenizer_stories stories of the training subset. Validation: the whole validation file."""
    files = {entry['name']: directory / entry['name'] for entry in SOURCE['files']}
    train_path = next(path for name, path in files.items() if 'train' in name)
    valid_path = next(path for name, path in files.items() if 'valid' in name)
    train, tail, total = [], deque(maxlen=audit_stories), 0
    for story in stories(train_path, drop_leading_fragment=False):
        if len(train) < train_stories: train.append(story)
        else: tail.append(story)
        total += 1
    if total < train_stories + audit_stories: raise SystemExit('The training file has too few stories for these subset sizes.')
    audit = list(tail)
    valid = list(stories(valid_path, drop_leading_fragment=True))
    output.mkdir(parents=True, exist_ok=True)
    subsets = dict(train=train, audit=audit, validation=valid, tokenizer=train[:tokenizer_stories])
    for name, items in subsets.items():
        (output / f'{name}-stories.json').write_text(json.dumps(items, ensure_ascii=False))
    record = dict(source=dict(dataset=SOURCE['dataset'], revision=SOURCE['revision'], files={e['name']: e['sha256'] for e in SOURCE['files']}),
                  rules=dict(train=f'first {train_stories} stories of the training file', audit=f'last {audit_stories} stories of the training file (never trained on)',
                             tokenizer=f'first {tokenizer_stories} stories of the training subset', validation='every story of the validation file after its leading fragment',
                             normalization='split on <|endoftext|>, CRLF to LF, strip surrounding whitespace, drop empty stories'),
                  storiesInTrainingFile=total,
                  subsets={name: dict(stories=len(items), characters=sum(map(len, items)), sha256=digest(items)) for name, items in subsets.items()})
    (output / 'prepared.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def load(name, output=OUTPUT):
    """name: train, audit, validation, or tokenizer."""
    return json.loads((output / f'{name}-stories.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-stories', type=int, default=DEFAULTS['train_stories'])
    parser.add_argument('--audit-stories', type=int, default=DEFAULTS['audit_stories'])
    parser.add_argument('--tokenizer-stories', type=int, default=DEFAULTS['tokenizer_stories'])
    arguments = parser.parse_args()
    fetch()
    result = prepare(arguments.train_stories, arguments.audit_stories, arguments.tokenizer_stories)
    print(json.dumps(result['subsets'], indent=2))
