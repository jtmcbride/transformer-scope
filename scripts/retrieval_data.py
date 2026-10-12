"""Exhaustive, unambiguous associative retrieval; split assignment precedes training."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations, product

SYMBOLS = ['<BOS>', 'A', 'B', 'C', 'D', '0', '1', '2', '3', '=', ';', '?']
KEYS = range(1, 5)
VALUES = range(5, 9)
SPLIT_SALT = 'retrieval-v1-seed-2026'

@dataclass(frozen=True)
class Example:
    tokens: tuple[int, ...]
    target: int

    @property
    def text(self):
        return ' '.join(SYMBOLS[token] for token in self.tokens[1:])


def examples(bindings=3):
    """Unique keys, independent values (repeats allowed), query never the last binding."""
    for keys in permutations(KEYS, bindings):
        for values in product(VALUES, repeat=bindings):
            prefix = [0]
            for key, value in zip(keys, values):
                prefix.extend((key, 9, value, 10))
            for query_index in range(bindings - 1):
                yield Example(tuple(prefix + [11, keys[query_index], 9]), values[query_index])


def partition(example):
    bucket = int.from_bytes(sha256((SPLIT_SALT + ':' + ','.join(map(str, example.tokens))).encode()).digest()[:8], 'big') % 100
    return 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'


def dataset_digest(items):
    payload = '\n'.join(','.join(map(str, item.tokens)) + ':' + str(item.target) for item in items)
    return sha256(payload.encode()).hexdigest()
