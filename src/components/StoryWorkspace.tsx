import { createContext, useCallback, useContext, useEffect, useState, useRef, useMemo, type ReactNode } from 'react';
import type { ModelManifest } from '../engine/contracts';
import { StoryTokenInspector } from './StoryTokenInspector';
import { selectForkToken } from '../story-fork';
import { mergeStoryNode, nodeRoute, parseWorkspace, type StoryWorkspace, type StoryRoot, type StoryNode, type SharedStoryGroup } from '../story-workspace';
import type { BpeDocument } from '../engine/bpe';
import { StoryHeatmapComparison } from './StoryHeatmapComparison';
import { useStoryLibrary } from './useStoryLibrary';
import { EditStoryName, StoryHistory } from './StoryHistory';
import { WorkspaceLibrary, downloadWorkspaceData } from './WorkspaceLibrary';
import { workspaceName } from '../story-workspace-names';
import '../story-history.css';
interface PairScanRequest { nodeId: string; branch: 'original' | 'edited'; index: number; targetTokenId: number }
interface Store { analysis: { heads: boolean; pairs: boolean }; setAnalysisOpen: (kind: 'heads' | 'pairs', open: boolean) => void; openPairScan: (request: PairScanRequest) => void; data: StoryWorkspace; saveRoot: (root: StoryRoot) => void; saveNode: (node: StoryNode) => void; navigate: (id: string) => void; focus: (id: string) => void; reset: (remount?: boolean) => boolean; heatmapBusy: boolean; setInspectorScanning: (value: boolean) => void }
const Context = createContext<Store | null>(null);
export function useStoryWorkspace() { return useContext(Context); }
export function StoryWorkspaceProvider({ manifest, tokenizer, inputKey, children }: { manifest: ModelManifest; tokenizer: BpeDocument; inputKey: string; children: (data: StoryWorkspace, navigation: number, tools: ReactNode) => ReactNode }) {
  const shelf = useStoryLibrary(manifest), { data, update: setData } = shelf;
  const [navigation, setNavigation] = useState(0);
  const history = useRef<HTMLHeadingElement>(null), transition = useRef(0);
  const [heatmapBusy,setHeatmapBusy] = useState(false), [inspectorScanning,setInspectorScanning] = useState(false);
  const [pairScan, setPairScan] = useState<PairScanRequest | null>(null);
  const [analysis, setAnalysis] = useState({ heads: false, pairs: false });
  const setAnalysisOpen = useCallback((kind: 'heads' | 'pairs', open: boolean) => setAnalysis(old => old[kind] === open ? old : { ...old, [kind]: open }), []);
  const scanNode = data.nodes.find(node => node.id === pairScan?.nodeId);
  const scanSelection = useMemo(() => pairScan && scanNode ? selectForkToken(scanNode.plan, scanNode.frame, pairScan.branch, pairScan.index) : null, [scanNode, pairScan]);
  const input = useRef(inputKey);
  function clearView() { ++transition.current; setPairScan(null); setAnalysis({ heads: false, pairs: false }); setHeatmapBusy(false); setInspectorScanning(false); }
  function reset(remount = true) { if (!shelf.begin()) return false; clearView(); if (remount) setNavigation(n => n + 1); return true; }
  useEffect(() => { if (input.current !== inputKey && reset()) input.current = inputKey; }, [inputKey]);
  const saveRoot = useCallback((root: StoryRoot) => setData(old => ({ ...old, root })), [setData]);
  const saveNode = useCallback((node: StoryNode) => setData(old => ({ ...old, nodes: old.nodes.some(n => n.id === node.id) ? old.nodes.map(n => n.id === node.id ? mergeStoryNode(n, node) : n) : [...old.nodes, node], route: [...nodeRoute(old.nodes, node.parentId ?? ''), node.id] })), [setData]);
  const saveShared = useCallback((group: SharedStoryGroup, nodes: [StoryNode,StoryNode]) => setData(old => {
    const byId = new Map(nodes.map(node => [node.id,node]));
    return { ...old, nodes:[...old.nodes.map(node => byId.has(node.id) ? mergeStoryNode(node, byId.get(node.id)!) : node),...nodes.filter(node => !old.nodes.some(saved => saved.id === node.id))],
      sharedForks:old.sharedForks?.some(saved => saved.id === group.id) ? old.sharedForks.map(saved => saved.id === group.id ? group : saved) : [...(old.sharedForks ?? []),group] };
  }),[setData]);
  function focus(id: string) { setData(old => ({ ...old, route: nodeRoute(old.nodes, id) })); }
  function navigate(id: string) { setPairScan(null); setData(old => ({ ...old, route: nodeRoute(old.nodes, id) })); setNavigation(n => n + 1); }
  function openSaved(id: string) { if (!shelf.open(id)) return false; clearView(); setNavigation(n => n + 1); return true; }
  function rename(id: string | null, name?: string) {
    setData(old => {
      const named = <T extends { name?: string }>(value: T): T => { const next = { ...value }; if (name) next.name = name; else delete next.name; return next; };
      return id === null ? named(old) : { ...old, nodes: old.nodes.map(node => node.id === id ? named(node) : node) };
    });
  }
  async function upload(file?: File) {
    if (!file) return;
    const generation = ++transition.current;
    try { if (file.size > 10 * 1024 * 1024) throw new Error('Workspace files must be smaller than 10 MB.'); const next = parseWorkspace(JSON.parse(await file.text()), manifest); if (generation !== transition.current) return; if (shelf.begin(next)) { clearView(); setNavigation(n => n + 1); } }
    catch (e) { if (generation === transition.current) shelf.setError(`Could not open this workspace: ${String(e)}`); }
  }
  const tools = <div className="workspace-toolbar" aria-label="Current story workspace"><div><small>Workspace</small><strong>{workspaceName(data)}</strong><span role="status">{data.root ? shelf.saved ? 'Saved on this browser' : 'Not saved yet' : 'Create a story to begin'}</span></div>
      {data.root && <button type="button" onClick={() => { history.current?.focus({ preventScroll: true }); history.current?.scrollIntoView({ block: 'start' }); }}>Branch history · {data.nodes.length}</button>}
      <WorkspaceLibrary library={shelf.library} manifest={manifest} onOpen={openSaved} onRemove={shelf.remove} />
    </div>;
  return <Context.Provider value={{ analysis, setAnalysisOpen, openPairScan: request => { navigate(request.nodeId); setAnalysisOpen('pairs', true); setPairScan(request); }, data, saveRoot, saveNode, navigate, focus, reset, heatmapBusy, setInspectorScanning }}>
    {shelf.error && <p role="alert" className="error">{shelf.error}</p>}
    {shelf.storageError && <p role="alert" className="error">{shelf.storageError}</p>}
    {input.current !== inputKey && <p className="workspace-pending-prompt">The new prompt is ready. Your previous workspace is still open; resolve the library error before starting another.<button type="button" onClick={() => { if (reset()) input.current = inputKey; }}>Start new prompt workspace</button></p>}
    {pairScan && scanNode && scanSelection ? <>{tools}<StoryTokenInspector key={`${pairScan.nodeId}:${pairScan.branch}:${pairScan.index}`} selection={scanSelection}
      manifest={manifest} tokenizer={tokenizer} initialTargetTokenId={pairScan.targetTokenId} parentId={scanNode.id} depth={nodeRoute(data.nodes, scanNode.id).length}
      startStep={scanNode.plan.source.step} count={scanNode.frame[pairScan.branch]!.choices.length}
      onClose={() => navigate(scanNode.id)} onNavigate={index => setPairScan({ ...pairScan, index })} onForkStart={() => {}} /></> : children(data, navigation, tools)}
    <section className="story-workspace-map" aria-label="Story branch workspace">
      <div className="continuation-heading"><div><p className="eyebrow">Your paths through the story</p><h2 ref={history} tabIndex={-1}>Story workspace</h2></div><strong>{workspaceName(data)}</strong></div>
      <p>Follow a path to revisit its tokens or try another change. Previous workspaces stay in Saved workspaces above. Downloads include every comparison in this workspace.</p>
      <div className="continuation-actions"><button type="button" disabled={!data.root} onClick={() => downloadWorkspaceData(data)}>Download workspace</button><label className="workspace-open">Open workspace<input type="file" accept=".json,application/json" aria-label="Open story workspace" onChange={event => { void upload(event.target.files?.[0]); event.target.value = ''; }} /></label>
        <EditStoryName key={shelf.activeId} value={data.name} suggested={workspaceName(data)} label="workspace" disabled={!data.root} onSave={name => rename(null, name)} />
      </div>
      {data.root && <StoryHistory data={data} tokenizer={tokenizer} onNavigate={navigate} onRename={rename} />}
    </section>
    {data.root && <StoryHeatmapComparison key={shelf.activeId} workspace={data} manifest={manifest} tokenizer={tokenizer} disabled={inspectorScanning} onBusy={setHeatmapBusy} onOpenStory={navigate} onSaveShared={saveShared} />}

  </Context.Provider>;
}
