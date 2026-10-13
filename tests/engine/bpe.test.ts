import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { BpeTokenizer, validateBpe } from '../../src/engine/bpe';
import document from '../fixtures/text-tokenizer.json';
import parity from '../fixtures/bpe-parity.json';

const tokenizer = new BpeTokenizer(validateBpe(document, 2048));

it('encodes exactly as the independent Python tokenizer, including edge cases, and round-trips', () => {
  expect(parity.cases.length).toBe(163);
  for (const { text, ids } of parity.cases) {
    expect(tokenizer.encode(text), JSON.stringify(text.slice(0, 40))).toEqual(ids);
    expect(tokenizer.decode(ids)).toBe(text);
  }
  expect(tokenizer.encode('Lily', { bos: false, eos: true }).at(-1)).toBe(1);
});

it('rejects malformed tokenizers and unpaired surrogates', () => {
  const copy = () => JSON.parse(readFileSync('tests/fixtures/text-tokenizer.json', 'utf8'));
  expect(() => validateBpe(copy(), 4096)).toThrow('vocabulary');
  expect(() => validateBpe({ ...copy(), pattern: '\\w+' }, 2048)).toThrow('Unsupported');
  const forward = copy(); forward.merges[0] = [300, 4];
  expect(() => validateBpe(forward, 2048)).toThrow('earlier merges');
  expect(() => tokenizer.encode('bad \uD800 text')).toThrow('unpaired surrogate');
  expect(tokenizer.encode('ok 🙂').length).toBeGreaterThan(1);
});
