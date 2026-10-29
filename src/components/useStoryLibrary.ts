import { useCallback, useEffect, useRef, useState } from 'react';
import type { ModelManifest } from '../engine/contracts';
import { emptyWorkspace, type StoryWorkspace } from '../story-workspace';
import { activeWorkspace, loadLibrary, openWorkspace, persistLibrary, removeWorkspace, startWorkspace, updateWorkspace, workspaceLibraryKey, workspaceStorageKey, type WorkspaceLibrary } from '../story-workspace-library';

export function useStoryLibrary(manifest: ModelManifest) {
  const key = workspaceLibraryKey(manifest);
  const [initial] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      return { ...loadLibrary(raw, localStorage.getItem(workspaceStorageKey(manifest)), manifest), raw };
    } catch (e) { return { ...loadLibrary(null, null, manifest), raw: null, error: `Could not read the browser library: ${String(e)}` }; }
  });
  const [library, setLibrary] = useState(initial.library);
  const [error, setError] = useState(initial.error), [storageError, setStorageError] = useState('');
  const [persisted, setPersisted] = useState<WorkspaceLibrary | null>(null);
  const current = useRef(library), raw = useRef(initial.raw);
  current.current = library;
  const persist = useCallback((next: WorkspaceLibrary) => {
    try {
      raw.current = persistLibrary(localStorage, key, raw.current, next);
      setPersisted(next); setStorageError(''); return true;
    } catch (e) {
      setStorageError(`Automatic saving is unavailable or storage is full. Download the workspace to keep it. ${String(e)}`);
      setPersisted(null); return false;
    }
  }, [key]);
  useEffect(() => { persist(library); }, [library, persist]);
  useEffect(() => {
    const changed = (event: StorageEvent) => {
      if (event.storageArea === localStorage && (event.key === key || event.key === null) && event.newValue !== raw.current) {
        setPersisted(null); setStorageError('The saved library changed in another tab. Download your current workspace, then reload to reopen the saved library.');
      }
    };
    window.addEventListener('storage', changed);
    return () => window.removeEventListener('storage', changed);
  }, [key]);
  const activeId = library.activeId;
  // A response from a disposed story can never write into the newly active workspace.
  const update = useCallback((transform: (data: StoryWorkspace) => StoryWorkspace) => {
    setLibrary(old => old.activeId === activeId ? updateWorkspace(old, transform) : old);
  }, [activeId]);
  function replace(next: WorkspaceLibrary) {
    if (!persist(next)) return false;
    current.current = next; setLibrary(next); setError(''); return true;
  }
  function begin(data = emptyWorkspace(manifest)) {
    try { return replace(startWorkspace(current.current, data)); }
    catch (e) { setError(`Could not start another workspace: ${String(e)}`); return false; }
  }
  function open(entryId: string) {
    try { return replace(openWorkspace(current.current, entryId, manifest)); }
    catch (e) { setError(`Could not open this workspace: ${String(e)}`); return false; }
  }
  function remove(entryId: string) {
    try { return replace(removeWorkspace(current.current, entryId)); }
    catch (e) { setError(`Could not remove this workspace: ${String(e)}`); return false; }
  }
  return { library, data: activeWorkspace(library), activeId, update, begin, open, remove, saved: persisted === library, error, storageError, setError };
}
