"""Byte-level BPE: a fully specified tokenizer that the browser reproduces exactly.

Specification (format 2):
- Pre-split text with PATTERN (leftmost alternative wins; Python `re` and JavaScript with the u flag agree on it),
  then encode each piece as UTF-8 bytes.
- Token IDs: 0 <BOS>, 1 <EOS>, 2 <PAD>; 3 + b for byte b; 259 + i for merge i.
- Training: repeatedly merge the most frequent adjacent pair, counting each piece by its frequency; ties go to the
  smallest (left ID, right ID). Merges never cross piece boundaries.
- Encoding a piece: repeatedly take the adjacent pair with the lowest merge rank and merge all its non-overlapping
  occurrences from left to right, until no adjacent pair has a rank.
- Decoding concatenates the bytes of non-special tokens and decodes UTF-8 strictly.

Usage: .venv/bin/python scripts/text_tokenizer.py   (trains on .cache/text/tokenizer-stories.json, writes .cache/text/bpe.json)
"""
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path

PATTERN = r" ?[A-Za-z]+| ?[0-9]| ?[^A-Za-z0-9 \t\n\r]+|[ \t\n\r]+"
SPECIALS = ['<BOS>', '<EOS>', '<PAD>']
BYTE_OFFSET = len(SPECIALS)
VOCABULARY_SIZE = 2048
splitter = re.compile(PATTERN)


def pieces(text):
    return [match.encode('utf-8') for match in splitter.findall(text)]


def merge_sequence(sequence, pair, new_id):
    """Merge every non-overlapping occurrence of pair, scanning left to right."""
    merged, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            merged.append(new_id); i += 2
        else:
            merged.append(sequence[i]); i += 1
    return merged


def train(texts, vocabulary_size=VOCABULARY_SIZE):
    counts = Counter(piece for text in texts for piece in pieces(text))
    words = [[BYTE_OFFSET + byte for byte in piece] for piece in counts]
    frequencies = list(counts.values())
    pair_counts, where = Counter(), {}
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequencies[index]
            where.setdefault(pair, set()).add(index)
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while BYTE_OFFSET + 256 + len(merges) < vocabulary_size:
        while heap:
            count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -count and count < 0: break
        else:
            raise ValueError('Ran out of pairs before reaching the vocabulary size.')
        new_id = BYTE_OFFSET + 256 + len(merges)
        merges.append(pair)
        changed = set()
        for index in where.pop(pair, ()):
            word = words[index]
            if not any(a == pair[0] and b == pair[1] for a, b in zip(word, word[1:])): continue
            for old in zip(word, word[1:]):
                pair_counts[old] -= frequencies[index]; changed.add(old)
            words[index] = word = merge_sequence(word, pair, new_id)
            for new in zip(word, word[1:]):
                pair_counts[new] += frequencies[index]; changed.add(new)
                where.setdefault(new, set()).add(index)
        for changed_pair in changed:
            if pair_counts[changed_pair] > 0: heapq.heappush(heap, (-pair_counts[changed_pair], changed_pair))
            else: pair_counts.pop(changed_pair, None)
    return merges


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.bytes = [b''] * BYTE_OFFSET + [bytes([b]) for b in range(256)]
        for a, b in self.merges: self.bytes.append(self.bytes[a] + self.bytes[b])
        self.cache = {}

    def encode_piece(self, piece):
        if piece in self.cache: return self.cache[piece]
        sequence = [BYTE_OFFSET + byte for byte in piece]
        while len(sequence) > 1:
            ranked = [(self.ranks[pair], pair) for pair in zip(sequence, sequence[1:]) if pair in self.ranks]
            if not ranked: break
            rank, pair = min(ranked)
            sequence = merge_sequence(sequence, pair, BYTE_OFFSET + 256 + rank)
        self.cache[piece] = sequence
        return sequence

    def encode(self, text, bos=True, eos=False):
        ids = [0] if bos else []
        for piece in pieces(text): ids.extend(self.encode_piece(piece))
        return ids + ([1] if eos else [])

    def decode(self, ids):
        return b''.join(self.bytes[i] for i in ids if i >= BYTE_OFFSET).decode('utf-8')

    def display(self, token_id):
        if token_id < BYTE_OFFSET: return SPECIALS[token_id]
        try: return self.bytes[token_id].decode('utf-8')
        except UnicodeDecodeError: return ''.join(f'<0x{byte:02X}>' for byte in self.bytes[token_id])

    def document(self):
        return dict(schemaVersion=2, kind='byte-bpe', pattern=PATTERN, bosTokenId=0, eosTokenId=1, padTokenId=2, byteOffset=BYTE_OFFSET,
                    merges=[list(pair) for pair in self.merges], tokens=[self.display(i) for i in range(len(self.bytes))])

    def serialized(self):
        return (json.dumps(self.document(), indent=1, ensure_ascii=False) + '\n').encode('utf-8')


def load(path):
    document = json.loads(Path(path).read_text())
    assert document['schemaVersion'] == 2 and document['kind'] == 'byte-bpe' and document['pattern'] == PATTERN
    return Tokenizer(document['merges'])


if __name__ == '__main__':
    import time
    from text_data import OUTPUT, load as load_stories
    start = time.perf_counter()
    tokenizer = Tokenizer(train(load_stories('tokenizer')))
    payload = tokenizer.serialized()
    (OUTPUT / 'bpe.json').write_bytes(payload)
    print(f'Trained {len(tokenizer.merges)} merges in {time.perf_counter() - start:.1f} s; sha256 {hashlib.sha256(payload).hexdigest()}')
