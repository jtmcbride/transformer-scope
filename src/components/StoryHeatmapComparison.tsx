import { useEffect, useMemo, useRef, useState } from 'react';
import type { ModelManifest } from '../engine/contracts';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import { tokenLabel } from '../engine/tokenizer';
import { selectionEdits } from '../continuation';
import type { SharedStoryGroup, StoryNode, StoryWorkspace } from '../story-workspace';
import { checkpointKey } from '../story-workspace';
import { pairEffect, type PairEffectMetric, compareInteractions, rankInteractionChanges, scanStoryHeatmaps, workspaceStoryPaths } from '../story-heatmap-comparison';
import { headPairHeatmap, headPairKey } from '../head-pair-heatmap';
import { pairScanTotal, pairStrengthOptions, type HeadPairScanRow, type PairStrengths } from '../story-head-pair-scan';
import { ModelClient } from '../workers/client';
import { loadStoryModel, withStoryPrefix } from '../workers/story-runtime';
import { InteractionMatrix } from './InteractionMatrix';
import { SharedStoryBranches, type SharedBranchDraft } from './SharedStoryBranches';
import '../story-heatmap-comparison.css';

type Pick = { path: string; index: number };
type Scan = { key: string; rows: [HeadPairScanRow[],HeadPairScanRow[]]; busy: boolean; status: string; error: string };
interface Props { workspace: StoryWorkspace; manifest: ModelManifest; tokenizer: BpeDocument; disabled: boolean; onBusy: (value: boolean) => void; onOpenStory: (id: string) => void; onSaveShared: (group: SharedStoryGroup, nodes: [StoryNode,StoryNode]) => void }
const points = (n: number) => `${n > 0 ? '+' : ''}${(n*100).toFixed(4)} pp`;
const EMPTY: [HeadPairScanRow[],HeadPairScanRow[]] = [[],[]];
function Scale({limit,label}:{limit:number;label:string}) {
  return <div className="pair-map-legend" aria-label={label} data-color-limit={limit}>{limit ? <><div className="pair-map-ramp" aria-hidden="true"/><div><span>−{(limit*100).toFixed(4)} pp</span><span>0</span><span>+{(limit*100).toFixed(4)} pp</span></div></> : <p>Zero range until a nonzero measured value appears.</p>}</div>;
}
export function StoryHeatmapComparison({ workspace, manifest, tokenizer, disabled, onBusy, onOpenStory, onSaveShared }: Props) {
  const [open,setOpen] = useState(false), [picks,setPicks] = useState<[Pick,Pick]>([{ path:'',index:0 },{ path:'',index:0 }]);
  const [strengths,setStrengths] = useState<PairStrengths>('mute'), [assignment,setAssignment] = useState(''), [order,setOrder] = useState<'absolute'|'increase'|'decrease'>('absolute');
  const [targetMode,setTargetMode] = useState<'generated'|'fixed'>('generated'), [chosenToken,setChosenToken] = useState(() => workspace.root?.frame.original.choices[0]?.tokenId ?? 1), [metric,setMetric] = useState<PairEffectMetric>('interaction');
  const targetTokenId = targetMode === 'fixed' ? chosenToken : undefined;
  const target = useMemo(() => targetMode === 'fixed' ? {kind:'fixed',tokenId:chosenToken} : {kind:'generated'}, [targetMode,chosenToken]);
  const probabilityMode = metric === 'probability', effectName = probabilityMode ? 'probability change' : 'interaction';
  const effect = (row: HeadPairScanRow) => pairEffect(row,metric);
  const vocabularyOptions = useMemo(() => tokenizer.tokens.map((_,id) => id !== 0 && id !== 2 && <option key={id} value={id}>{tokenLabel(tokenizer,id)} · token {id}</option>), [tokenizer]);
  const [scan,setScan] = useState<Scan | null>(null), [selectedKey,setSelectedKey] = useState('');
  const [forkBusy,setForkBusy] = useState(false);
  const owner = useRef<ModelClient | null>(null), epoch = useRef(0);
  const sharedPanel = useRef<HTMLDivElement>(null);
  const paths = useMemo(() => workspaceStoryPaths(workspace),[workspace.root,workspace.nodes]);
  const chosen = useMemo(() => { const preferred = paths.slice(-2); return picks.map((pick,i) => { const path = paths.find(path => path.id === pick.path) ?? preferred[i] ?? preferred[0]; return path ? { path, index:Math.min(pick.index,path.count-1) } : null; }); },[paths,picks]);
  const sources = useMemo(() => chosen.map(pick => pick?.path.select(pick.index)),[chosen]);
  const tokenOptions = useMemo(() => chosen.map(pick => pick ? Array.from({length:pick.path.count},(_,index) => ({index,step:pick.path.start+index,label:tokenLabel(tokenizer,pick.path.select(index).choice.tokenId)})) : []),[chosen,tokenizer]);
  const inputKey = useMemo(() => JSON.stringify([chosen.map(pick => pick ? [pick.path.id,pick.index] : null),sources,strengths,target]),[chosen,sources,strengths,target]);
  const current = scan?.key === inputKey ? scan : null, busy = !!current?.busy, rows = current?.rows ?? EMPTY;
  const working = busy || forkBusy;
  useEffect(() => {onBusy(working);},[working,onBusy]);
  const total = pairScanTotal(manifest.config,strengths);
  const options = pairStrengthOptions.find(option => option.value === strengths)!.factors;
  const [rowFactor,columnFactor] = options.find(pair => pair.join(':') === assignment) ?? options[0];
  const encoder = useMemo(() => new BpeTokenizer(tokenizer),[tokenizer]);
  useEffect(() => () => { ++epoch.current; owner.current?.dispose(); owner.current = null; onBusy(false); },[inputKey,onBusy]);
  function stop() { ++epoch.current; owner.current?.dispose(); owner.current = null; onBusy(false); setScan(old => old ? { ...old,busy:false,status:'Comparison stopped. Results are partial.' } : null); }
  async function start() {
    if (working || disabled || !sources[0] || !sources[1]) return;
    const captured = structuredClone(sources) as [NonNullable<typeof sources[0]>,NonNullable<typeof sources[1]>];
    setPicks(chosen.map(pick => ({ path:pick!.path.id,index:pick!.index })) as [Pick,Pick]); setSelectedKey('');
    const generation = ++epoch.current, valid = () => epoch.current === generation;
    setScan({ key:inputKey,rows:[[],[]],busy:true,status:'Scanning both tokens…',error:'' }); onBusy(true);
    const client = new ModelClient(); owner.current = client;
    try {
      await loadStoryModel(client,manifest); if (!valid()) return;
      await scanStoryHeatmaps(captured,manifest.config,strengths,(source,use) => withStoryPrefix(client,source.prefix,use),(side,row) => {
        if (valid()) setScan(old => old?.key === inputKey ? { ...old,rows:side === 0 ? [[...old.rows[0],row],old.rows[1]] : [old.rows[0],[...old.rows[1],row]] } : old);
      },() => !valid(),targetTokenId);
      if (valid()) setScan(old => old ? { ...old,busy:false,status:'Heatmap comparison complete.' } : old);
    } catch(e) { if (valid()) setScan(old => old ? { ...old,busy:false,status:'Comparison failed. Results are partial.',error:String(e) } : old); }
    finally { client.dispose(); if (valid()) { owner.current = null; onBusy(false); } }
  }
  const maps = useMemo(() => rows.map(side => headPairHeatmap(side,manifest.config,rowFactor,columnFactor)),[rows,manifest.config,rowFactor,columnFactor]);
  const changes = useMemo(() => compareInteractions(rows[0],rows[1],metric),[rows,metric]);
  const changeByKey = useMemo(() => new Map(changes.map(row => [row.key,row])),[changes]);
  const ranked = useMemo(() => rankInteractionChanges(changes,order),[changes,order]);
  const limit = rows.flat().reduce((max,row) => Math.max(max,Math.abs(effect(row))),0), differenceLimit = changes.reduce((max,row) => Math.max(max,Math.abs(row.difference)),0);
  const selected = rows.map(side => side.find(row => headPairKey(row.edits) === selectedKey));
  const pair = selected[0] ?? selected[1];
  const pairVisible = maps.some(map => map.cells.some(line => line.some(row => row && headPairKey(row.edits) === selectedKey)));
  function swap() {
    if (working || !chosen[0] || !chosen[1]) return;
    const nextPicks: [Pick,Pick] = [{ path:chosen[1].path.id,index:chosen[1].index },{ path:chosen[0].path.id,index:chosen[0].index }];
    const nextKey = JSON.stringify([nextPicks.map(pick => [pick.path,pick.index]),[sources[1],sources[0]],strengths,target]);
    setPicks(nextPicks);
    if (current) setScan({ ...current,key:nextKey,rows:[rows[1],rows[0]] });
  }
  function download() {
    if (!current) return;
    const data = { format:'transformer-story-heatmap-comparison',version:targetMode === 'fixed' || probabilityMode ? 2 : 1,target,metric,backend:'cpu',differenceDefinition:probabilityMode ? '(p_pair_B - p_source_B) - (p_pair_A - p_source_A)' : 'interaction_B - interaction_A',checkpoint:checkpointKey(manifest),strengths,sources,labels:chosen.map(pick => pick!.path.label),rows,complete:current.status === 'Heatmap comparison complete.' && rows.every(side => side.length === total),status:current.status,error:current.error || undefined,difference:'B minus A',changes };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const a = document.createElement('a'); a.href=url; a.download='story-heatmap-comparison.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url),0);
  }
  return <details className="story-map-comparison" open={open} onToggle={event => { const next=event.currentTarget.open; setOpen(next); if (!next && busy) stop(); }}>
    <summary>Compare head-pair scans across story tokens</summary>
    {open && <section aria-label="Compare story interaction heatmaps">
      <h3>{targetMode === 'fixed' ? 'Which pairs affect the same token differently?' : 'What changed between these two predictions?'}</h3><p>Choose a token from each story path. Both scans use the same head strengths, but each keeps its own exact starting text, recorded token, edits, and sampling draw.</p>
      {paths.length === 0 ? <p>Generate story tokens to begin. Saved branch comparisons will appear here too.</p> : <>
        <div className="story-map-picks">{chosen.map((pick,i) => pick && <article key={i} aria-label={`Token ${i === 0 ? 'A' : 'B'} selection`}>
          <h4>Token {i === 0 ? 'A' : 'B'}</h4><label>Story path<select aria-label={`Story ${i === 0 ? 'A' : 'B'} path`} value={pick.path.id} disabled={forkBusy} onChange={event => { const path=event.target.value; setPicks(old => old.map((p,j) => j === i ? {path,index:0} : {path:chosen[j]!.path.id,index:chosen[j]!.index}) as [Pick,Pick]); setSelectedKey(''); }}>{paths.map(path => <option key={path.id} value={path.id}>{path.label}</option>)}</select></label>
          <label>Generated token<select aria-label={`Token ${i === 0 ? 'A' : 'B'} position`} value={pick.index} disabled={forkBusy} onChange={event => { const index=Number(event.target.value); setPicks(old => old.map((p,j) => ({path:chosen[j]!.path.id,index:j === i ? index:chosen[j]!.index})) as [Pick,Pick]); setSelectedKey(''); }}>{tokenOptions[i].map(option => <option key={option.index} value={option.index}>Step {option.step} · {option.label}</option>)}</select></label>
          <p data-testid={`comparison-token-${i}`}><strong>{tokenLabel(tokenizer,sources[i]!.choice.tokenId)}</strong> · step {sources[i]!.step} · recorded probability {(sources[i]!.choice.probability*100).toFixed(4)}%</p>
          <details><summary>Starting text and captured settings</summary><p className="inspection-prefix">{encoder.preview(sources[i]!.prefix)}</p><p>{selectionEdits(sources[i]!).map(edit => `Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`).join('; ') || 'No head edits.'}</p><p>Temperature {sources[i]!.settings.temperature} · top-p {sources[i]!.settings.topP} · seed {sources[i]!.settings.seed} · global step {sources[i]!.step}.</p></details>
        </article>)}</div>
        <div className="continuation-actions"><label>Probability target<select aria-label="Comparison target mode" value={targetMode} disabled={forkBusy} onChange={event => { setTargetMode(event.target.value as typeof targetMode); setSelectedKey(''); if (event.target.value === 'fixed') setMetric('probability'); }}><option value="generated">Each recorded generated token</option><option value="fixed">Same chosen token in both stories</option></select></label>
          {targetMode === 'fixed' && <label>Chosen next token<select aria-label="Shared scan target token" value={chosenToken} disabled={forkBusy} onChange={event => {setChosenToken(Number(event.target.value));setSelectedKey('');}}>{vocabularyOptions}</select></label>}
          <label>Measure<select aria-label="Compare head-pair effects" value={metric} disabled={forkBusy} onChange={event => setMetric(event.target.value as PairEffectMetric)}><option value="probability">Combined probability change</option><option value="interaction">Head-pair interaction</option></select></label>
          <label>Scan strengths<select aria-label="Comparison scan strengths" value={strengths} disabled={forkBusy} onChange={event => {setStrengths(event.target.value as PairStrengths);setAssignment('');setSelectedKey('');}}>{pairStrengthOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
          <button type="button" disabled={working || disabled} onClick={() => void start()}>Scan both tokens</button><button type="button" disabled={!busy} onClick={stop}>Stop map comparison</button><button type="button" disabled={working} onClick={swap}>Swap A and B</button><button type="button" disabled={working || !current} onClick={download}>Download heatmap comparison</button>
          {options.length > 1 && <label>Axis strengths<select aria-label="Comparison heatmap strength assignment" value={`${rowFactor}:${columnFactor}`} onChange={event => setAssignment(event.target.value)}>{options.map(([a,b]) => <option key={`${a}:${b}`} value={`${a}:${b}`}>Rows {a}× · Columns {b}×</option>)}</select></label>}
        </div>
        {targetMode === 'fixed' && <p data-testid="comparison-fixed-target">Both scans measure <strong>{tokenLabel(tokenizer,chosenToken)}</strong> (token {chosenToken}). The recorded tokens above remain the actual generated choices. Changing the target clears unfinished work and old results.</p>}
        <p role="status" data-testid="comparison-map-status">{current?.status ?? 'Ready to compare maps.'} A: {rows[0].length} / {total} · B: {rows[1].length} / {total} · {changes.length} matching combinations.</p>
        {busy && <progress value={rows[0].length+rows[1].length} max={2*total} aria-label="Two-token scan progress" />}
        {current?.error && <p role="alert">Could not finish map comparison: {current.error}</p>}
        <p className="hint">A and B share a symmetric color range of ±{(limit*100).toFixed(4)} pp across all measured strengths. Blue is negative {effectName}; orange is positive. {targetMode === 'fixed' ? 'The target token is the same in both stories. Each probability change is relative to that story’s own captured settings.' : 'Different generated tokens are different probability targets.'} The starting text and captured edits can still differ; this does not isolate one cause.</p>
        <Scale limit={limit} label={`Shared A and B ${effectName} color scale`} />
        <div className="story-comparison-maps">{maps.map((map,i) => <article key={i} aria-label={`Token ${i === 0 ? 'A' : 'B'} ${effectName} map`} data-color-limit={limit}><h4>{i === 0 ? 'A' : 'B'} · {tokenLabel(tokenizer,targetTokenId ?? sources[i]!.choice.tokenId)} · step {sources[i]!.step}</h4>
          <InteractionMatrix cells={map.cells.map(line => line.map(row => row ? {key:headPairKey(row.edits),value:effect(row),description:`Combined change ${points(row.delta)}.`} : undefined))} config={manifest.config} rowFactor={rowFactor} columnFactor={columnFactor} limit={limit} selectedKey={selectedKey} disabled={working} onSelect={setSelectedKey} valueLabel={effectName} />
          <p>{map.measured} / {map.total} combinations in this strength view.</p></article>)}</div>
        <div className="story-comparison-difference"><article aria-label={probabilityMode ? 'Probability effect difference map' : 'Interaction difference map'}><h4>Difference · B minus A</h4>
          <InteractionMatrix cells={maps[0].cells.map(line => line.map(row => { const change=row && changeByKey.get(headPairKey(row.edits));return change ? {key:change.key,value:change.difference,description:`A ${points(effect(change.a))}; B ${points(effect(change.b))}.`}:undefined; }))} config={manifest.config} rowFactor={rowFactor} columnFactor={columnFactor} limit={differenceLimit} selectedKey={selectedKey} disabled={working} onSelect={setSelectedKey} valueLabel={`Difference in ${effectName} (B minus A)`} />
          <Scale limit={differenceLimit} label={`Difference ${effectName} color scale`} />
          <p>Difference color range: ±{(differenceLimit*100).toFixed(4)} pp. {probabilityMode ? 'Positive means the pair encourages the chosen token more (or suppresses it less) in B than A; negative means the reverse. This subtracts the two changes, not the two final probabilities.' : 'Positive means more positive interaction in B; negative means more negative.'} Only pairs measured in both maps appear here. Hatched cells are unmeasured, not zero.</p></article>
          <aside aria-label="Compared pair details" aria-live="polite">{pair ? <><h4>Selected pair</h4><p>{pair.edits.map(edit => `Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`).join(' + ')}</p>
            <dl>{selected.map((row,i) => <div key={i}><dt>{i === 0 ? 'A' : 'B'} {effectName}</dt><dd data-testid={`compared-interaction-${i}`}>{row ? points(effect(row)) : 'Not measured'}</dd><dt>{i === 0 ? 'A' : 'B'} source → with pair</dt><dd data-testid={`compared-probabilities-${i}`}>{row ? `${(row.before*100).toFixed(4)}% → ${(row.after*100).toFixed(4)}%` : 'Not measured'}</dd><dt>{i === 0 ? 'A' : 'B'} combined probability change</dt><dd>{row ? points(row.delta) : 'Not measured'}</dd></div>)}<div><dt>Difference in {effectName} · B minus A</dt><dd data-testid="compared-interaction-difference">{selected[0] && selected[1] ? points(effect(selected[1])-effect(selected[0])) : 'Not measured'}</dd></div></dl>
            {!pairVisible && <p>Selected pair is outside the current strength view.</p>}
            <button type="button" disabled={working || !selected[0] || !selected[1]} onClick={() => {sharedPanel.current?.scrollIntoView({block:'start'});sharedPanel.current?.querySelector<HTMLElement>('select')?.focus({preventScroll:true});}}>Set up both continuations</button>
            <p>Branch both stories with this pair below, or open a source story to inspect its computation.</p>{chosen.map((pick,i) => <button key={i} type="button" disabled={working} onClick={() => onOpenStory(pick!.path.nodeId ?? '')}>Open story {i === 0 ? 'A' : 'B'}</button>)}
          </> : <><h4>Select a cell in any map</h4><p>The same head pair is highlighted across A, B, and the difference map. The selected values keep their original signs.</p></>}</aside>
        </div>
        <div ref={sharedPanel}><SharedStoryBranches key={inputKey} workspace={workspace} manifest={manifest} tokenizer={tokenizer} disabled={busy || disabled} onBusy={setForkBusy} onSave={onSaveShared} onOpenStory={onOpenStory}
          draft={selected[0] && selected[1] ? {sources:sources as SharedBranchDraft['sources'],parents:chosen.map(pick => pick!.path.nodeId) as SharedBranchDraft['parents'],labels:chosen.map(pick => pick!.path.label) as SharedBranchDraft['labels'],edits:selected[0].edits,targetTokenId,effects:[selected[0],selected[1]]} : undefined} /></div>
        <label>Rank differences<select aria-label={probabilityMode ? 'Rank probability effect differences' : 'Rank interaction differences'} value={order} onChange={event => setOrder(event.target.value as typeof order)}><option value="absolute">Largest change (absolute)</option><option value="increase">Most positive B minus A</option><option value="decrease">Most negative B minus A</option></select></label>
        <ol aria-label={probabilityMode ? 'Largest probability effect differences' : 'Largest interaction changes'}>{ranked.slice(0,8).map(row => <li key={row.key} data-pair-key={row.key}><button type="button" disabled={working} onClick={() => {setSelectedKey(row.key);setAssignment(row.a.edits.map(edit => edit.factor).join(':'));}}>{row.a.edits.map(edit => `L${edit.layer+1} H${edit.head+1} ${edit.factor}×`).join(' + ')} · A {points(effect(row.a))} → B {points(effect(row.b))} · change {points(row.difference)}</button></li>)}</ol>
        <p className="hint">{probabilityMode && 'Probability effect = probability with both head changes minus probability with source settings. Difference = effect in B minus effect in A. '}Interaction = combined probability change − sum of individual changes, relative to each token's own captured settings. Equal-strength maps mirror pairs; mixed maps show both assignments. Rankings cover only the chosen scope and matching completed trials. Results are temporary; download them to keep this comparison. Saved story branches remain in the workspace.</p>
      </>}
    </section>}
  </details>;
}
