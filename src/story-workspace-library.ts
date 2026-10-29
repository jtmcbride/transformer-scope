import type { ModelManifest } from './engine/contracts';
import { requireInput } from './engine/contracts';
import { emptyWorkspace, parseWorkspace, type StoryWorkspace } from './story-workspace';

export interface WorkspaceEntry { id: string; updatedAt: number; data: unknown; recovered?: boolean }
export interface WorkspaceLibrary { format: 'transformer-story-workspace-library'; version: 1; activeId: string; entries: WorkspaceEntry[] }
export const workspaceStorageKey = (manifest: ModelManifest) => `transformer-story-workspace:${manifest.modelId}`;
export const workspaceLibraryKey = (manifest: ModelManifest) => `${workspaceStorageKey(manifest)}:library`;
const id = (): string => crypto.randomUUID();

export function activeWorkspace(library: WorkspaceLibrary): StoryWorkspace {
  // Only loadLibrary, startWorkspace and openWorkspace establish the active entry.
  return library.entries.find(entry => entry.id === library.activeId)!.data as StoryWorkspace;
}
function parseLibrary(raw: string): WorkspaceLibrary {
  const value = JSON.parse(raw) as WorkspaceLibrary;
  requireInput(value?.format === 'transformer-story-workspace-library' && value.version === 1, 'Unsupported workspace library.');
  requireInput(Array.isArray(value.entries) && value.entries.length > 0 && value.entries.length <= 1000, 'Invalid workspace library entries.');
  const seen = new Set<string>();
  for (const entry of value.entries) {
    requireInput(!!entry && typeof entry.id === 'string' && entry.id.length > 0 && entry.id.length <= 100 && !seen.has(entry.id), 'Invalid saved workspace ID.');
    requireInput(Number.isFinite(entry.updatedAt) && entry.updatedAt >= 0 && Object.hasOwn(entry, 'data') && (entry.recovered === undefined || typeof entry.recovered === 'boolean'), 'Invalid saved workspace.');
    seen.add(entry.id);
  }
  requireInput(typeof value.activeId === 'string' && seen.has(value.activeId), 'Missing active workspace.');
  return value;
}
/** The envelope keeps unreadable and older-checkpoint data downloadable, without running it. */
export function loadLibrary(raw: string | null, legacy: string | null, manifest: ModelManifest, makeId = id, now = Date.now()): { library: WorkspaceLibrary; error: string } {
  let library: WorkspaceLibrary;
  try {
    if (raw !== null) library = parseLibrary(raw);
    else {
      const entryId = makeId();
      library = { format: 'transformer-story-workspace-library', version: 1, activeId: entryId, entries: [{ id: entryId, updatedAt: now, data: legacy === null ? emptyWorkspace(manifest) : JSON.parse(legacy) }] };
    }
  } catch (e) {
    const recoveredId = makeId(), activeId = makeId();
    const backup: WorkspaceEntry[] = [];
    if (raw !== null && legacy !== null) {
      try { backup.push({ id: makeId(), updatedAt: now, data: JSON.parse(legacy) }); }
      catch { backup.push({ id: makeId(), updatedAt: now, data: legacy, recovered: true }); }
    }
    return { library: { format: 'transformer-story-workspace-library', version: 1, activeId, entries: [
      { id: recoveredId, updatedAt: now, data: raw ?? legacy, recovered: true },
      ...backup,
      { id: activeId, updatedAt: now, data: emptyWorkspace(manifest) },
    ] }, error: `Could not reopen the saved data. Its original contents are kept in Saved workspaces for download. ${String(e)}` };
  }
  try {
    const data = parseWorkspace(activeWorkspace(library), manifest);
    return { library: { ...library, entries: library.entries.map(entry => entry.id === library.activeId ? { ...entry, data } : entry) }, error: '' };
  } catch (e) {
    const activeId = makeId();
    return { library: { ...library, activeId, entries: [...library.entries, { id: activeId, updatedAt: now, data: emptyWorkspace(manifest) }] }, error: `Could not reopen the last workspace. It is kept in Saved workspaces for download. ${String(e)}` };
  }
}
export function updateWorkspace(library: WorkspaceLibrary, update: (data: StoryWorkspace) => StoryWorkspace, now = Date.now()): WorkspaceLibrary {
  return { ...library, entries: library.entries.map(entry => entry.id === library.activeId ? { ...entry, updatedAt: now, data: update(entry.data as StoryWorkspace) } : entry) };
}
export function startWorkspace(library: WorkspaceLibrary, data: StoryWorkspace, entryId = id(), now = Date.now()): WorkspaceLibrary {
  // Empty drafts need no history entry; actual stories and recovered data are never evicted.
  const entries = library.entries.filter(entry => entry.id !== library.activeId || activeWorkspace(library).root !== null);
  requireInput(entries.length < 1000, 'The library has reached 1,000 workspaces. Download and remove an older workspace before starting another.');
  return { ...library, activeId: entryId, entries: [...entries, { id: entryId, updatedAt: now, data }] };
}
export function openWorkspace(library: WorkspaceLibrary, entryId: string, manifest: ModelManifest): WorkspaceLibrary {
  const entry = library.entries.find(entry => entry.id === entryId);
  requireInput(!!entry && !entry.recovered, 'Download this recovered data to inspect it.');
  const data = parseWorkspace(entry.data, manifest);
  return { ...library, activeId: entryId, entries: library.entries.map(item => item.id === entryId ? { ...item, data } : item) };
}
export function removeWorkspace(library: WorkspaceLibrary, entryId: string): WorkspaceLibrary {
  requireInput(entryId !== library.activeId, 'Open another workspace before removing this one.');
  return { ...library, entries: library.entries.filter(entry => entry.id !== entryId) };
}
/** One write keeps the previous stories and new active selection together. No quota eviction. */
export function persistLibrary(storage: Pick<Storage, 'getItem' | 'setItem'>, key: string, expected: string | null, library: WorkspaceLibrary): string {
  requireInput(storage.getItem(key) === expected, 'The saved library changed in another tab. Download your current workspace, then reload to reopen the saved library.');
  const raw = JSON.stringify(library);
  storage.setItem(key, raw);
  return raw;
}
