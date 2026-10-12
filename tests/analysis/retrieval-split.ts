import { createHash } from 'node:crypto';

/** Mirrors scripts/retrieval_data.py: generator order, targets, and the hashed split rule. */
export interface RetrievalExample { tokens: number[]; target: number }
const salt = 'retrieval-v1-seed-2026';

function* permutations(items: number[], size: number, prefix: number[] = []): Generator<number[]> {
  if (prefix.length === size) { yield prefix; return; }
  for (const item of items) if (!prefix.includes(item)) yield* permutations(items, size, [...prefix, item]);
}
function* product(items: number[], repeat: number, prefix: number[] = []): Generator<number[]> {
  if (prefix.length === repeat) { yield prefix; return; }
  for (const item of items) yield* product(items, repeat, [...prefix, item]);
}
export function retrievalExamples(bindings = 3): RetrievalExample[] {
  const examples: RetrievalExample[] = [];
  for (const keys of permutations([1, 2, 3, 4], bindings)) for (const values of product([5, 6, 7, 8], bindings)) {
    const prefix = [0];
    keys.forEach((key, index) => prefix.push(key, 9, values[index], 10));
    for (let query = 0; query < bindings - 1; query++) examples.push({ tokens: [...prefix, 11, keys[query], 9], target: values[query] });
  }
  return examples;
}
export function partition(example: RetrievalExample): 'train' | 'validation' | 'test' {
  const bucket = createHash('sha256').update(`${salt}:${example.tokens.join(',')}`).digest().readBigUInt64BE(0) % 100n;
  return bucket < 70n ? 'train' : bucket < 85n ? 'validation' : 'test';
}
export function datasetDigest(items: RetrievalExample[]): string {
  return createHash('sha256').update(items.map(item => `${item.tokens.join(',')}:${item.target}`).join('\n')).digest('hex');
}
