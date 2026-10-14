"""Standard-library deployment audit of exact model identities and binary payloads."""
import argparse
import hashlib
import json
import math
import struct
from text_tokenizer import Tokenizer, PATTERN
from pathlib import Path


def audit(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    canonical = dict(manifest)
    revision = canonical.pop('modelRevision')
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == revision, f'{directory}: revision mismatch'
    assert manifest['architectureVersion'] == 'preln-absolute-tied-v1'
    assert manifest['dtype'] == 'float32-le' and manifest['matrixOrientation'] == 'input-output'
    binary = (directory / manifest['weightsFile']).read_bytes()
    tokenizer_bytes = (directory / manifest['tokenizerFile']).read_bytes()
    assert hashlib.sha256(binary).hexdigest() == manifest['weightsSha256'], f'{directory}: weights checksum mismatch'
    assert hashlib.sha256(tokenizer_bytes).hexdigest() == manifest['tokenizerSha256'], f'{directory}: tokenizer checksum mismatch'
    offset = 0
    names = set()
    for entry in manifest['tensors']:
        assert entry['name'] not in names and entry['byteOffset'] == offset
        assert all(isinstance(size, int) and size > 0 for size in entry['shape'])
        assert entry['byteLength'] == math.prod(entry['shape']) * 4
        names.add(entry['name'])
        offset += entry['byteLength']
    assert offset == len(binary)
    assert all(math.isfinite(value) for (value,) in struct.iter_unpack('<f', binary)), f'{directory}: non-finite weight values'
    tokenizer = json.loads(tokenizer_bytes)
    symbols = tokenizer['tokens']
    assert len(symbols) == manifest['config']['vocabularySize']
    if tokenizer['kind'] == 'byte-bpe':
        assert tokenizer['schemaVersion'] == 2 and tokenizer['pattern'] == PATTERN
        assert [tokenizer[key] for key in ['bosTokenId', 'eosTokenId', 'padTokenId', 'byteOffset']] == [0, 1, 2, 3]
        assert 259 + len(tokenizer['merges']) == len(symbols)
        for rank, pair in enumerate(tokenizer['merges']):
            assert len(pair) == 2 and all(type(value) is int and 3 <= value < 259 + rank for value in pair)
        assert len({tuple(pair) for pair in tokenizer['merges']}) == len(tokenizer['merges'])
        assert Tokenizer(tokenizer['merges']).document()['tokens'] == symbols
    else:
        assert tokenizer['kind'] == 'symbols' and len(set(symbols)) == len(symbols)
        assert all(isinstance(symbol, str) and symbol and not any(char.isspace() for char in symbol) for symbol in symbols)
    assert 0 <= tokenizer['bosTokenId'] < len(symbols)
    assert (directory / 'probe.json').stat().st_size > 0
    if directory.name == 'tinystories':
        assert json.loads((directory / 'evaluation.json').read_text())['modelRevision'] == revision
    if directory.name == 'retrieval':
        for filename in ['recipes.json', 'evaluation.json']:
            assert json.loads((directory / filename).read_text())['modelRevision'] == revision
    return manifest['modelId'], len(binary), revision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=Path('public/models'))
    args = parser.parse_args()
    for name in ['verification', 'retrieval', 'tinystories']:
        identity, size, revision = audit(args.models / name)
        print(f'{identity}: {size} bytes, verified revision {revision}')
