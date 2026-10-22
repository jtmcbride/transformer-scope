import { createContext, useCallback, useContext, useEffect, useState, useRef, type ReactNode } from 'react';
import type { ModelManifest } from '../engine/contracts';
import { emptyWorkspace, nodeRoute, parseWorkspace, type StoryWorkspace, type StoryRoot, type StoryNode } from '../story-workspace';
interface Store { data: StoryWorkspace; saveRoot: (root: StoryRoot) => void; saveNode: (node: StoryNode) => void; navigate: (id: string) => void; focus: (id: string) => void; reset: (remount?: boolean) => void }
const Context = createContext<Store | null>(null);
export function useStoryWorkspace() { return useContext(Context); }
export function StoryWorkspaceProvider({ manifest, inputKey, children }: { manifest: ModelManifest; inputKey: string; children: (data: StoryWorkspace, navigation: number) => ReactNode }) {
  const storageKey = `transformer-story-workspace:${manifest.modelId}`;
  const [initial] = useState(() => { try { const raw = localStorage.getItem(storageKey); return { data: raw ? parseWorkspace(JSON.parse(raw), manifest) : emptyWorkspace(manifest), error: '' }; } catch (e) { return { data: emptyWorkspace(manifest), error: `Could not reopen the saved workspace: ${String(e)}` }; } });
  const [data, setData] = useState(initial.data), [navigation, setNavigation] = useState(0), [error, setError] = useState(initial.error), [saved, setSaved] = useState(false);
  const input = useRef(inputKey);
  function reset(remount = true) { setData(emptyWorkspace(manifest)); try { localStorage.removeItem(storageKey); } catch { /* Export remains available. */ } if (remount) setNavigation(n => n + 1); }
  useEffect(() => { if (input.current !== inputKey) { input.current = inputKey; reset(); } }, [inputKey]);
  const saveRoot = useCallback((root: StoryRoot) => setData(old => ({ ...old, root })), []);
  const saveNode = useCallback((node: StoryNode) => setData(old => ({ ...old, nodes: old.nodes.some(n => n.id === node.id) ? old.nodes.map(n => n.id === node.id ? node : n) : [...old.nodes, node], route: [...nodeRoute(old.nodes, node.parentId ?? ''), node.id] })), []);
  function focus(id: string) { setData(old => ({ ...old, route: nodeRoute(old.nodes, id) })); }
  function navigate(id: string) { setData(old => ({ ...old, route: nodeRoute(old.nodes, id) })); setNavigation(n => n + 1); }
  useEffect(() => {
    setSaved(false);
    // Keep invalid/incompatible stored files recoverable until a real experiment is created or imported.
    if (!data.root) return;
    try { localStorage.setItem(storageKey, JSON.stringify(data)); setSaved(true); }
    catch { setError('Automatic saving is unavailable or storage is full. Download the workspace to keep it.'); }
  }, [data, storageKey]);
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = 'story-workspace.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  async function upload(file?: File) {
    if (!file) return;
    try { if (file.size > 10 * 1024 * 1024) throw new Error('Workspace files must be smaller than 10 MB.'); const next = parseWorkspace(JSON.parse(await file.text()), manifest); setData(next); setError(''); setNavigation(n => n + 1); }
    catch (e) { setError(`Could not open this workspace: ${String(e)}`); }
  }
  return <Context.Provider value={{ data, saveRoot, saveNode, navigate, focus, reset }}>
    <section className="story-workspace-map" aria-label="Story branch workspace">
      <div className="continuation-heading"><div><p className="eyebrow">Your paths through the story</p><h2>Story workspace</h2></div><span role="status">{saved ? 'Saved on this browser' : data.root ? 'Not saved yet' : 'Create a story to begin'}</span></div>
      <p>Keep comparisons, revisit their tokens, and explore another change. Downloads include the whole workspace. Run input or Reset story starts a new workspace; download first to keep this experiment.</p>
      <div className="continuation-actions"><button type="button" disabled={!data.root} onClick={download}>Download workspace</button><label className="workspace-open">Open workspace<input type="file" accept=".json,application/json" aria-label="Open story workspace" onChange={event => { void upload(event.target.files?.[0]); event.target.value = ''; }} /></label></div>
      {error && <p role="alert" className="error">{error}</p>}
      {data.root && <nav aria-label="Story branch map"><button type="button" aria-current={data.route.length === 0 ? 'location' : undefined} onClick={() => navigate('')}>Source story</button>
        {data.nodes.map((node, index) => <button type="button" key={node.id} style={{ marginLeft: `${Math.min(nodeRoute(data.nodes, node.id).length, 6) * 12}px` }} aria-current={data.route.at(-1) === node.id ? 'location' : undefined} onClick={() => navigate(node.id)}>
          <strong>Comparison {index + 1}</strong><span>Step {node.plan.source.step} · Layer {node.plan.edit.layer + 1}, Head {node.plan.edit.head + 1} · {node.plan.edit.factor.toFixed(2)}× · {node.plan.duration === 'prediction' ? 'once' : 'onward'}</span><small>{node.parentId ? `From comparison ${data.nodes.findIndex(n => n.id === node.parentId) + 1}` : 'From source story'} · {node.plan.source.branch === 'original' ? 'source settings' : 'changed path'}</small>
        </button>)}
      </nav>}
    </section>
    {children(data, navigation)}
  </Context.Provider>;
}
