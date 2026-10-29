import { expect, it } from 'vitest';
import { activeWorkspace, loadLibrary, openWorkspace, persistLibrary, removeWorkspace, startWorkspace, updateWorkspace } from '../../src/story-workspace-library';
import { emptyWorkspace, parseWorkspace, type StoryWorkspace } from '../../src/story-workspace';
import { branchName, workspaceName } from '../../src/story-workspace-names';
import { loadShippedArtifact } from './retrieval-artifact';
import fixture from '../fixtures/story-trajectory-workspace.json';

const { manifest, tokenizer } = loadShippedArtifact('tinystories');
const story = () => structuredClone(fixture.workspace) as unknown as StoryWorkspace;
const ids = () => { let next = 0; return () => `workspace-${++next}`; };

it('migrates the old workspace intact and preserves it when a new prompt becomes active', () => {
  const original = story(); original.name = 'The red ball'; original.nodes[0].name = 'Ask a friend';
  const { library, error } = loadLibrary(null, JSON.stringify(original), manifest, ids(), 10);
  expect(error).toBe(''); expect(activeWorkspace(library)).toEqual(original);
  const next = startWorkspace(library, emptyWorkspace(manifest), 'next', 20);
  expect(next.entries).toHaveLength(2); expect(next.entries[0].data).toEqual(original);
  expect(activeWorkspace(next).root).toBeNull();
  const reopened = openWorkspace(next, library.activeId, manifest);
  expect(activeWorkspace(reopened)).toEqual(original);
  expect(loadLibrary(JSON.stringify(reopened), null, manifest).library).toEqual(reopened);
  expect(library.entries).toHaveLength(1);
});

it('updates just the active workspace and keeps every real history without eviction', () => {
  let library = loadLibrary(null, JSON.stringify(story()), manifest, ids()).library;
  const initial = structuredClone(library.entries[0]);
  for (let i = 0; i < 35; i++) library = startWorkspace(library, { ...story(), name: `Story ${i}` }, `new-${i}`, i);
  const changed = updateWorkspace(library, data => ({ ...data, name: 'Current story' }), 100);
  expect(changed.entries).toHaveLength(36); expect(changed.entries[0]).toEqual(initial);
  expect(activeWorkspace(changed).name).toBe('Current story');
  expect(activeWorkspace(library).name).toBe('Story 34');
  const removed = removeWorkspace(changed, initial.id);
  expect(removed.entries).toHaveLength(35); expect(changed.entries).toHaveLength(36);
  expect(() => removeWorkspace(changed, changed.activeId)).toThrow('Open another');
});

it('replaces empty drafts while retaining real and unreadable saved work', () => {
  const raw = '{broken library';
  const loaded = loadLibrary(raw, null, manifest, ids(), 10);
  expect(loaded.error).toContain('original contents');
  const next = startWorkspace(loaded.library, emptyWorkspace(manifest), 'fresh', 20);
  expect(next.entries).toHaveLength(2);
  expect(next.entries[0]).toMatchObject({ data: raw, recovered: true });
  expect(() => openWorkspace(next, next.entries[0].id, manifest)).toThrow('Download');
  expect(loadLibrary(JSON.stringify(next), null, manifest).library.entries[0].data).toBe(raw);
});

it('retains incompatible and invalid legacy data for download without executing it', () => {
  for (const source of ['{not json', JSON.stringify({ ...story(), checkpoint: 'older-checkpoint' })]) {
    const loaded = loadLibrary(null, source, manifest, ids(), 5);
    expect(loaded.error).not.toBe(''); expect(activeWorkspace(loaded.library).root).toBeNull();
    expect(loaded.library.entries).toHaveLength(2);
    const stored = loaded.library.entries[0];
    expect(stored.data).toEqual(source.startsWith('{not') ? source : JSON.parse(source));
  }
});

it('writes one complete library atomically and fails without evicting anything on quota or stale-tab conflicts', () => {
  const original = loadLibrary(null, JSON.stringify(story()), manifest, ids()).library;
  let raw: string | null = null, writes = 0;
  const storage = { getItem: () => raw, setItem: (_key: string, value: string) => { writes++; raw = value; } };
  const saved = persistLibrary(storage, 'library', null, original);
  const next = startWorkspace(original, emptyWorkspace(manifest), 'new');
  const fail = { ...storage, setItem: () => { throw new Error('Quota exceeded'); } };
  expect(() => persistLibrary(fail, 'library', saved, next)).toThrow('Quota exceeded');
  expect(JSON.parse(raw!)).toEqual(original); expect(writes).toBe(1);
  raw = 'changed by another tab';
  expect(() => persistLibrary(storage, 'library', saved, next)).toThrow('another tab');
  expect(raw).toBe('changed by another tab'); expect(writes).toBe(1);
});

it('validates optional names without changing model inputs, frames, ancestry or shared links', () => {
  const original = story(), renamed = story();
  renamed.name = 'A ball and a friend'; renamed.nodes[0].name = 'Ask instead of run';
  const parsed = parseWorkspace(JSON.parse(JSON.stringify(renamed)), manifest);
  expect(parsed).toEqual(renamed);
  expect(parsed.root).toEqual(original.root); expect(parsed.sharedForks).toEqual(original.sharedForks);
  expect(parsed.nodes.map(({ name: _name, ...node }) => node)).toEqual(original.nodes);
  for (const name of ['', '  ', 'x'.repeat(81), 42, null]) {
    expect(() => parseWorkspace({ ...renamed, name }, manifest)).toThrow('Names');
    const bad = story(); bad.nodes[0].name = name as string;
    expect(() => parseWorkspace(bad, manifest)).toThrow('Names');
  }
});

it('describes the actual chosen token and head changes, while honoring custom names', () => {
  const data = story(), node = data.nodes[0];
  const text = branchName(node, tokenizer as import('../../src/engine/bpe').BpeDocument);
  expect(text).toContain('L1/H1'); expect(text).toContain('at “');
  const changed = structuredClone(node); changed.name = 'A surprising turn';
  expect(branchName(changed, tokenizer as import('../../src/engine/bpe').BpeDocument)).toBe('A surprising turn');
  expect(workspaceName({ ...data, name: 'My experiment' })).toBe('My experiment');
  expect(workspaceName(emptyWorkspace(manifest))).toBe('New story');
});
