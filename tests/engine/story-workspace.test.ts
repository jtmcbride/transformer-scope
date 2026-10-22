import { expect, it } from 'vitest';
import { emptyWorkspace, parseWorkspace, nodeRoute, type StoryWorkspace } from '../../src/story-workspace';
import { StoryContinuation } from '../../src/continuation';
import { StoryFork } from '../../src/story-fork';
import { loadShippedArtifact } from './retrieval-artifact';
import reference from '../fixtures/story-forks-reference.json';
import nested from '../fixtures/nested-story-forks-reference.json';
import root from '../fixtures/continuation-reference.json';
it('round trips a comparison tree and resumes exact prefixes and sampling offsets', async () => {
  const { manifest } = loadShippedArtifact('tinystories');
  const w = emptyWorkspace(manifest);
  const session = new StoryContinuation(root.promptTokenIds, root.settings, 64, root.edit as any);
  session.restore({ step: 16, original: { ...root.original, finish: 'token limit' }, edited: { ...root.edited, finish: 'token limit' } });
  w.root = { prompt: root.prompt, tokens: root.promptTokenIds, settings: root.settings, edit: root.edit as any, frame: session.frame };
  // Original-branch selection is independent of the root's optional edited branch.
  const first = structuredClone(reference.cases[0]);
  const a = new StoryFork(first.plan as any, manifest.config);
  a.restore({ step: 12, original: { ...first.original, finish: 'token limit' }, edited: { ...first.edited, finish: 'token limit' } });
  w.nodes.push({ id: 'a', parentId: null, plan: a.plan, frame: a.frame });
  const b = new StoryFork(nested.cases[0].plan as any, manifest.config);
  b.restore({ step: 6, original: { ...nested.cases[0].original, finish: 'token limit' }, edited: { ...nested.cases[0].edited, finish: 'token limit' } });
  w.nodes.push({ id: 'b', parentId: 'a', plan: b.plan, frame: b.frame });
  w.route = nodeRoute(w.nodes, 'b');
  expect(parseWorkspace(JSON.parse(JSON.stringify(w)), manifest)).toEqual(w);
  expect(parseWorkspace({ ...w, version: 1 }, manifest)).toEqual(w);
  const bad = structuredClone(w); bad.nodes[1].plan.source.prefix[1] = 44;
  expect(() => parseWorkspace(bad, manifest)).toThrow('source');
  for (const corrupt of [(x: StoryWorkspace) => x.checkpoint = 'other', (x: StoryWorkspace) => x.nodes[0].parentId = 'b', (x: StoryWorkspace) => x.nodes[0].frame.original.choices[0].probability = NaN, (x: StoryWorkspace) => x.route = ['b']]) {
    const copy = structuredClone(w); corrupt(copy); expect(() => parseWorkspace(copy, manifest)).toThrow();
  }
  const pending = new StoryFork(first.plan as any, manifest.config);
  await pending.step(async () => [-9, -9, -9, 2]);
  const resumed = new StoryFork(pending.plan, manifest.config); resumed.restore(pending.frame);
  await pending.step(async () => [-9, -9, -9, 2]); await resumed.step(async () => [-9, -9, -9, 2]);
  expect(resumed.frame).toEqual(pending.frame);
});
it('migrates legacy workspaces and preserves validated two-head plans in version 2', async () => {
  const { manifest } = loadShippedArtifact('tinystories');
  const w = emptyWorkspace(manifest);
  const legacy = { ...w, version: 1 };
  expect(parseWorkspace(legacy, manifest)).toEqual(w);
  const session = new StoryContinuation(root.promptTokenIds, root.settings, 64, root.edit as any);
  session.restore({ step: 16, original: { ...root.original, finish: 'token limit' }, edited: { ...root.edited, finish: 'token limit' } });
  w.root = { prompt: root.prompt, tokens: root.promptTokenIds, settings: root.settings, edit: root.edit as any, frame: session.frame };
  const { default: pairs } = await import('../fixtures/story-head-pairs-reference.json');
  const example = pairs.cases[0];
  const fork = new StoryFork(example.plan as any, manifest.config);
  fork.restore({ step: 6, original: { ...example.original, finish: 'token limit' }, edited: { ...example.edited, finish: 'token limit' } });
  w.nodes.push({ id: 'pair', parentId: null, plan: fork.plan, frame: fork.frame }); w.route = ['pair'];
  expect(parseWorkspace(JSON.parse(JSON.stringify(w)), manifest)).toEqual(w);
  expect(() => parseWorkspace({ ...w, version: 1 }, manifest)).toThrow('version 2');
  w.nodes[0].plan.additionalEdits![0].head = w.nodes[0].plan.edit.head;
  w.nodes[0].plan.additionalEdits![0].layer = w.nodes[0].plan.edit.layer;
  expect(() => parseWorkspace(w, manifest)).toThrow('different heads');
});
