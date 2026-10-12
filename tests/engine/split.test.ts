import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { datasetDigest, partition, retrievalExamples } from '../analysis/retrieval-split';

it('regenerates the recorded retrieval split exactly', () => {
  const evaluation = JSON.parse(readFileSync('public/models/retrieval/evaluation.json', 'utf8')) as { split: { counts: Record<string, number>; digests: Record<string, string> } };
  const examples = retrievalExamples();
  expect(examples).toHaveLength(3072);
  for (const split of ['train', 'validation', 'test'] as const) {
    const members = examples.filter(example => partition(example) === split);
    expect(members.length, split).toBe(evaluation.split.counts[split]);
    expect(datasetDigest(members), split).toBe(evaluation.split.digests[split]);
  }
});
