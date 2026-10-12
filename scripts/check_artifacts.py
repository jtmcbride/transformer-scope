"""Standard-library deployment audit of exact model identities and binary payloads."""
import argparse
import hashlib
import json
import math
import struct
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
    assert len(symbols) == manifest['config']['vocabularySize'] and len(set(symbols)) == len(symbols)
    assert all(isinstance(symbol, str) and symbol and not any(char.isspace() for char in symbol) for symbol in symbols)
    assert 0 <= tokenizer['bosTokenId'] < len(symbols)
    assert (directory / 'probe.json').stat().st_size > 0
    if directory.name == 'retrieval':
        for filename in ['recipes.json', 'evaluation.json']:
            assert json.loads((directory / filename).read_text())['modelRevision'] == revision
    return manifest['modelId'], len(binary), revision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=Path('public/models'))
    args = parser.parse_args()
    for name in ['verification', 'retrieval']:
        identity, size, revision = audit(args.models / name)
        print(f'{identity}: {size} bytes, verified revision {revision}')
