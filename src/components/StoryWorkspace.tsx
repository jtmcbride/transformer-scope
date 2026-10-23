import { createContext, useCallback, useContext, useEffect, useState, useRef, useMemo, type ReactNode } from 'react';
import type { ModelManifest } from '../engine/contracts';
import { StoryTokenInspector } from './StoryTokenInspector';
import { selectForkToken, forkEditLabel } from '../story-fork';
import { emptyWorkspace, nodeRoute, parseWorkspace, type StoryWorkspace, type StoryRoot, type StoryNode, type SharedStoryGroup } from '../story-workspace';
import type { BpeDocument } from '../engine/bpe';
import { StoryHeatmapComparison } from './StoryHeatmapComparison';
interface PairScanRequest { nodeId: string; branch: 'original' | 'edited'; index: number; targetTokenId: number }
interface Store { openPairScan: (request: PairScanRequest) => void; data: StoryWorkspace; saveRoot: (root: StoryRoot) => void; saveNode: (node: StoryNode) => void; navigate: (id: string) => void; focus: (id: string) => void; reset: (remount?: boolean) => void; heatmapBusy: boolean; setInspectorScanning: (value: boolean) => void }
const Context = createContext<Store | null>(null);
export function useStoryWorkspace() { return useContext(Context); }
export function StoryWorkspaceProvider({ manifest, tokenizer, inputKey, children }: { manifest: ModelManifest; tokenizer: BpeDocument; inputKey: string; children: (data: StoryWorkspace, navigation: number) => ReactNode }) {
  const storageKey = `transformer-story-workspace:${manifest.modelId}`;
  const [initial] = useState(() => { try { const raw = localStorage.getItem(storageKey); return { data: raw ? parseWorkspace(JSON.parse(raw), manifest) : emptyWorkspace(manifest), error: '' }; } catch (e) { return { data: emptyWorkspace(manifest), error: `Could not reopen the saved workspace: ${String(e)}` }; } });
  const [data, setData] = useState(initial.data), [navigation, setNavigation] = useState(0), [error, setError] = useState(initial.error), [saved, setSaved] = useState(false);
  const [heatmapBusy,setHeatmapBusy] = useState(false), [inspectorScanning,setInspectorScanning] = useState(false);
  const [pairScan, setPairScan] = useState<PairScanRequest | null>(null);
  const scanNode = data.nodes.find(node => node.id === pairScan?.nodeId);
  const scanSelection = useMemo(() => pairScan && scanNode ? selectForkToken(scanNode.plan, scanNode.frame, pairScan.branch, pairScan.index) : null, [scanNode, pairScan]);
  const input = useRef(inputKey);
  function reset(remount = true) { setPairScan(null); setData(emptyWorkspace(manifest)); try { localStorage.removeItem(storageKey); } catch { /* Export remains available. */ } if (remount) setNavigation(n => n + 1); }
  useEffect(() => { if (input.current !== inputKey) { input.current = inputKey; reset(); } }, [inputKey]);
  const saveRoot = useCallback((root: StoryRoot) => setData(old => ({ ...old, root })), []);
  const saveNode = useCallback((node: StoryNode) => setData(old => ({ ...old, nodes: old.nodes.some(n => n.id === node.id) ? old.nodes.map(n => n.id === node.id ? node : n) : [...old.nodes, node], route: [...nodeRoute(old.nodes, node.parentId ?? ''), node.id] })), []);
  const saveShared = useCallback((group: SharedStoryGroup, nodes: [StoryNode,StoryNode]) => setData(old => {
    const byId = new Map(nodes.map(node => [node.id,node]));
    return { ...old, nodes:[...old.nodes.map(node => byId.get(node.id) ?? node),...nodes.filter(node => !old.nodes.some(saved => saved.id === node.id))],
      sharedForks:old.sharedForks?.some(saved => saved.id === group.id) ? old.sharedForks.map(saved => saved.id === group.id ? group : saved) : [...(old.sharedForks ?? []),group] };
  }),[]);
  function focus(id: string) { setData(old => ({ ...old, route: nodeRoute(old.nodes, id) })); }
  function navigate(id: string) { setPairScan(null); setData(old => ({ ...old, route: nodeRoute(old.nodes, id) })); setNavigation(n => n + 1); }
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
    try { if (file.size > 10 * 1024 * 1024) throw new Error('Workspace files must be smaller than 10 MB.'); const next = parseWorkspace(JSON.parse(await file.text()), manifest); setPairScan(null); setData(next); setError(''); setNavigation(n => n + 1); }
    catch (e) { setError(`Could not open this workspace: ${String(e)}`); }
  }
  return <Context.Provider value={{ openPairScan: request => { navigate(request.nodeId); setPairScan(request); }, data, saveRoot, saveNode, navigate, focus, reset, heatmapBusy, setInspectorScanning }}>
    <section className="story-workspace-map" aria-label="Story branch workspace">
      <div className="continuation-heading"><div><p className="eyebrow">Your paths through the story</p><h2>Story workspace</h2></div><span role="status">{saved ? 'Saved on this browser' : data.root ? 'Not saved yet' : 'Create a story to begin'}</span></div>
      <p>Keep comparisons, revisit their tokens, and explore another change. Downloads include the whole workspace. Run input or Reset story starts a new workspace; download first to keep this experiment.</p>
      <div className="continuation-actions"><button type="button" disabled={!data.root} onClick={download}>Download workspace</button><label className="workspace-open">Open workspace<input type="file" accept=".json,application/json" aria-label="Open story workspace" onChange={event => { void upload(event.target.files?.[0]); event.target.value = ''; }} /></label></div>
      {error && <p role="alert" className="error">{error}</p>}
      {data.root && <nav aria-label="Story branch map"><button type="button" aria-current={data.route.length === 0 ? 'location' : undefined} onClick={() => navigate('')}>Source story</button>
        {data.nodes.map((node, index) => <button type="button" key={node.id} style={{ marginLeft: `${Math.min(nodeRoute(data.nodes, node.id).length, 6) * 12}px` }} aria-current={data.route.at(-1) === node.id ? 'location' : undefined} onClick={() => navigate(node.id)}>
          <strong>Comparison {index + 1}</strong><span>Step {node.plan.source.step} · {forkEditLabel(node.plan)} · {node.plan.duration === 'prediction' ? 'once' : 'onward'}</span><small>{node.parentId ? `From comparison ${data.nodes.findIndex(n => n.id === node.parentId) + 1}` : 'From source story'} · {node.plan.source.branch === 'original' ? 'source settings' : 'changed path'}{data.sharedForks?.flatMap((group,i) => group.nodeIds.includes(node.id) ? [` · Shared ${i+1} · ${group.nodeIds[0] === node.id ? 'A' : 'B'}`] : [])}</small>
        </button>)}
      </nav>}
    </section>
    {data.root && <StoryHeatmapComparison workspace={data} manifest={manifest} tokenizer={tokenizer} disabled={inspectorScanning} onBusy={setHeatmapBusy} onOpenStory={navigate} onSaveShared={saveShared} />}
    {pairScan && scanNode && scanSelection ? <StoryTokenInspector key={`${pairScan.nodeId}:${pairScan.branch}:${pairScan.index}`} selection={scanSelection}
      manifest={manifest} tokenizer={tokenizer} initialTargetTokenId={pairScan.targetTokenId} parentId={scanNode.id} depth={nodeRoute(data.nodes, scanNode.id).length}
      startStep={scanNode.plan.source.step} count={scanNode.frame[pairScan.branch]!.choices.length}
      onClose={() => navigate(scanNode.id)} onNavigate={index => setPairScan({ ...pairScan, index })} onForkStart={() => {}} /> : children(data, navigation)}
  </Context.Provider>;
}
