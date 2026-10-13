import { readFileSync } from 'node:fs';
import { expect, test } from 'vitest';
import { BpeTokenizer, validateBpe } from '../../src/engine/bpe';

// Explicitly invoked (npm run parity:tokenizer) after scripts/export_text_tokenizer.py --all; needs local data in .cache/text.
test('TypeScript and Python encode every validation story identically', () => {
  const tokenizer = new BpeTokenizer(validateBpe(JSON.parse(readFileSync('.cache/text/bpe.json', 'utf8')), 2048));
  const cases = JSON.parse(readFileSync('.cache/text/validation-ids.json', 'utf8')) as { text: string; ids: number[] }[];
  let tokens = 0;
  for (const { text, ids } of cases) {
    const encoded = tokenizer.encode(text);
    if (encoded.length !== ids.length || encoded.some((id, i) => id !== ids[i])) expect(encoded, text.slice(0, 80)).toEqual(ids);
    if (tokenizer.decode(encoded) !== text) expect(tokenizer.decode(encoded)).toBe(text);
    tokens += ids.length;
  }
  console.info(`Identical token IDs and exact round trips for ${cases.length} stories (${tokens} tokens).`);
}, 600_000);
