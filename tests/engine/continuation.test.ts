import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { beforeAll, expect, it } from 'vitest';
import { chooseToken, drawAt, StoryContinuation, type Sampling } from '../../src/continuation';
import { BpeTokenizer } from '../../src/engine/bpe';
import { RunEngine } from '../../src/engine/runs';
import type { HeadScaleIntervention } from '../../src/engine/contracts';
import { loadShippedArtifact } from './retrieval-artifact';
import { tokenLabel } from '../../src/engine/tokenizer';
import reference from '../fixtures/continuation-reference.json';

const sampling: Sampling = { temperature: 1, topP: 1, seed: 42, maxTokens: 5 };
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
it('sampling masks BOS/PAD, keeps EOS, uses deterministic ties, and includes the top-p boundary', () => {
  expect(chooseToken([100, 2, 100, 2], { ...sampling, temperature: 0 }, .8).tokenId).toBe(1);
  // Eligible weights 1, .5, .25. Top-p .8 keeps the first two, then renormalizes to 2/3 and 1/3.
  const logits = [-100, 0, -100, Math.log(.5), Math.log(.25)];
  expect(chooseToken(logits, { ...sampling, topP: .8 }, .66).tokenId).toBe(1);
  expect(chooseToken(logits, { ...sampling, topP: .8 }, .67).tokenId).toBe(3);
  expect(chooseToken(logits, { ...sampling, topP: .1 }, .99).tokenId).toBe(1);
  expect(chooseToken(logits.map(x => x + 10000), sampling, .7).tokenId).toBe(chooseToken(logits, sampling, .7).tokenId);
  expect(() => chooseToken([0, NaN, 0, 1], sampling, 0)).toThrow('non-finite');
  expect(() => chooseToken(logits, { ...sampling, topP: 0 }, 0)).toThrow('Top-p');
  expect(Array.from({ length: 20 }, (_, i) => drawAt(0, i))).toEqual(Array.from({ length: 20 }, (_, i) => drawAt(0, i)));
  expect(drawAt(0, 0)).not.toBe(drawAt(42, 0));
});

it('paired steps are atomic on failure, retain exact IDs, and stop independently at EOS/context/token limits', async () => {
  const edit: HeadScaleIntervention = { id: 'test', kind: 'head.scale', layer: 0, head: 0, factor: 0, tokens: { kind: 'all' } };
  const story = new StoryContinuation([0, 3], { ...sampling, temperature: 0, maxTokens: 3 }, 5, edit);
  const letters = [-99, -10, -99, 3, 0];
  await expect(story.step(async (_, patch) => { if (patch) throw new Error('failed'); return letters; })).rejects.toThrow('failed');
  expect(story.frame.step).toBe(0);
  expect(story.frame.original.tokenIds).toEqual([0, 3]);
  const seen: number[][] = [];
  for (let i = 0; i < 3; i++) await story.step(async (ids, patch) => { seen.push(ids); return patch ? [-99, 5, -99, 0, 0] : letters; });
  expect(story.frame.original.tokenIds).toEqual([0, 3, 3, 3, 3]);
  expect(story.frame.original.finish).toBe('context limit');
  expect(story.frame.edited?.tokenIds).toEqual([0, 3, 1]);
  expect(story.frame.edited?.finish).toBe('end of story');
  expect(seen).toEqual([[0, 3], [0, 3], [0, 3, 3], [0, 3, 3, 3]]);
  expect(story.finished).toBe(true);
  await story.step(async () => { throw new Error('must not compute'); });
  const limit = new StoryContinuation([0], { ...sampling, maxTokens: 1, temperature: 0 }, 64);
  await limit.step(async () => letters);
  expect(limit.frame.original.finish).toBe('token limit');
  const full = new StoryContinuation([0, 3], sampling, 2);
  expect(full.finished).toBe(true);
  expect(() => new StoryContinuation([0], { ...sampling, seed: NaN }, 64)).toThrow('Seed');
});

it('partial UTF-8 is held for the next token and raw IDs survive noncanonical BPE boundaries', async () => {
  const { tokenizer } = loadShippedArtifact('tinystories');
  if (tokenizer.kind !== 'byte-bpe') throw new Error('Expected BPE');
  const bpe = new BpeTokenizer(tokenizer), bytes = Array.from(new TextEncoder().encode('🙂'), byte => byte + 3);
  expect(bpe.preview([0, ...bytes.slice(0, 2)])).toBe('');
  expect(bpe.preview([0, ...bytes])).toBe('🙂');
  expect(() => bpe.decode([0, ...bytes.slice(0, 2)])).toThrow();
  const withBom = '\uFEFFcafé';
  expect(bpe.decode(bpe.encode(withBom))).toBe(withBom);
  expect(bpe.preview(bpe.encode(withBom))).toBe(withBom);
  expect(tokenLabel(tokenizer, 3)).toBe('<0x00>');
  const story = new StoryContinuation([0, 65 + 3], { ...sampling, temperature: 0, maxTokens: 2 }, 64);
  const seen: number[][] = [];
  await story.step(async ids => { seen.push(ids); return Array.from({ length: 259 }, (_, id) => id === 66 + 3 ? 10 : -10); });
  await story.step(async ids => { seen.push(ids); return Array.from({ length: 259 }, (_, id) => id === 67 + 3 ? 10 : -10); });
  expect(seen).toEqual([[0, 68], [0, 68, 69]]);
});

it('sixteen original and edited autoregressive steps match the independent PyTorch rollout without retained tensors', async () => {
  const loaded = loadShippedArtifact('tinystories');
  expect(loaded.manifest.modelRevision).toBe(reference.modelRevision);
  const engine = new RunEngine(loaded.artifact), before = tf.memory().numTensors;
  const story = new StoryContinuation(reference.promptTokenIds, reference.settings, 64, reference.edit as HeadScaleIntervention);
  try {
    while (!story.finished) await story.step(async (tokenIds, edit) => {
      const base = await engine.runBaseline(tokenIds);
      let modified: string | undefined;
      try {
        const changed = edit ? await engine.compare({ baselineRunId: base.runId, interventions: [edit] }) : undefined;
        modified = changed?.modifiedRunId;
        return (await engine.inspectActivation(modified ?? base.runId, { stage: 'logits' }, { position: tokenIds.length - 1 })).values;
      } finally { if (modified) engine.releaseRun(modified); engine.releaseRun(base.runId); }
    });
    for (const key of ['original', 'edited'] as const) {
      const actual = story.frame[key]!;
      expect(actual.tokenIds).toEqual(reference[key].tokenIds);
      actual.choices.forEach((choice, i) => expect(choice.probability).toBeCloseTo(reference[key].choices[i].probability, 5));
    }
    expect(tf.memory().numTensors).toBe(before);
  } finally { engine.dispose(); }
}, 60_000);
