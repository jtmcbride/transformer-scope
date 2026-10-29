import { useRef, useState } from 'react';
import type { ModelManifest } from '../engine/contracts';
import { checkpointKey, type StoryWorkspace } from '../story-workspace';
import type { WorkspaceEntry, WorkspaceLibrary as Library } from '../story-workspace-library';
import { workspaceName } from '../story-workspace-names';

export function downloadWorkspaceData(data: unknown, filename = 'story-workspace.json', raw = false) {
  const content = raw && typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 0);
}
export function WorkspaceLibrary({ library, manifest, onOpen, onRemove }: { library: Library; manifest: ModelManifest; onOpen: (id: string) => boolean; onRemove: (id: string) => boolean }) {
  const [removing, setRemoving] = useState('');
  const summary = useRef<HTMLElement>(null);
  const entries = library.entries.filter(entry => entry.id !== library.activeId).sort((a, b) => b.updatedAt - a.updatedAt);
  function description(entry: WorkspaceEntry) {
    const data = entry.data as Partial<StoryWorkspace> | null;
    const shaped = data && typeof data === 'object' && data.format === 'transformer-story-workspace';
    const compatible = shaped && !entry.recovered && data.checkpoint === checkpointKey(manifest);
    const name = shaped && typeof data.name === 'string' && data.name.trim() ? data.name.slice(0, 80)
      : shaped && typeof data.root?.prompt === 'string' ? workspaceName({ ...data, name: undefined } as StoryWorkspace) : 'Recovered saved data';
    return { name, compatible, count: shaped && Array.isArray(data.nodes) ? data.nodes.length : null };
  }
  return <details className="workspace-library"><summary ref={summary}>Saved workspaces <span>{entries.length} previous {entries.length === 1 ? 'workspace' : 'workspaces'}</span></summary>
    <p>Starting a new prompt, resetting a story, or importing a file keeps the previous workspace here. Browser storage has limited space; download important work for a separate copy.</p>
    {entries.length === 0 ? <p>No previous workspaces yet. Your current story stays open above.</p> : <ul aria-label="Saved story workspaces">{entries.map(entry => {
      const { name, compatible, count } = description(entry);
      return <li key={entry.id} data-workspace-id={entry.id}><div><h3>{name}</h3><p>{count === null ? 'Recovered data' : `${count} ${count === 1 ? 'comparison' : 'comparisons'}`} · saved {new Date(entry.updatedAt).toLocaleString()}{!compatible && ' · download available for recovery'}</p></div>
        <div className="workspace-library-actions"><button type="button" disabled={!compatible} aria-label={`Open workspace: ${name}`} onClick={() => { if (onOpen(entry.id)) setRemoving(''); }}>Open</button>
          <button type="button" aria-label={`Download workspace: ${name}`} onClick={() => downloadWorkspaceData(entry.data, entry.recovered ? 'recovered-workspace-data.json' : 'story-workspace.json', entry.recovered)}>Download</button>
          <button type="button" aria-label={`Remove workspace: ${name}`} onClick={() => setRemoving(entry.id)}>Remove</button></div>
        {removing === entry.id && <div className="workspace-remove-confirm" role="group" aria-label={`Remove saved workspace: ${name}`}><p>Remove “{name}” and {count === null ? 'its saved contents' : `its ${count} comparisons`} from this browser? Download a copy first if you want to keep it.</p><button type="button" onClick={() => { if (onRemove(entry.id)) { setRemoving(''); summary.current?.focus(); } }}>Remove saved workspace</button><button type="button" onClick={() => { setRemoving(''); summary.current?.focus(); }}>Keep workspace</button></div>}
      </li>;
    })}</ul>}
  </details>;
}
