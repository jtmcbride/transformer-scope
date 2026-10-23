import { useEffect, useMemo, useRef, useState } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { ModelManifest } from '../engine/contracts';
import { tokenLabel } from '../engine/tokenizer';
import { selectionEdits, type StoryBranch, type StoryFrame, type StorySelection } from '../continuation';
import { firstForkDifference, forkEditLabel, proposedEdits, selectForkToken, StoryFork as Session, type ForkDuration, type ForkPlan } from '../story-fork';
import { ModelClient } from '../workers/client';
import { loadStoryModel, predictStory } from '../workers/story-runtime';
import { useStoryWorkspace } from './StoryWorkspace';
import type { ModelClient as AnalysisClient } from '../workers/client';
import { StoryHeadScan } from './StoryHeadScan';
import { StoryHeadPairScan } from './StoryHeadPairScan';
import type { HeadScaleIntervention } from '../engine/contracts';
import { BranchProbabilities } from './BranchProbabilities';
import { StoryTokenInspector } from './StoryTokenInspector';

interface Props { initialTargetTokenId?: number; source: StorySelection; manifest: ModelManifest; tokenizer: BpeDocument; head: { layer: number; head: number }; disabled: boolean; onStart: () => void; onInspectChange: (active: boolean) => void; depth: number; parentId?: string; analysisClient?: AnalysisClient; onChooseHead: (head: { layer: number; head: number }) => void }
export function StoryFork(props: Props) {
  const workspace = useStoryWorkspace();
  props = { ...props,disabled:props.disabled || !!workspace?.heatmapBusy };
  const [savedNode] = useState(() => {
    const nodes = workspace?.data.nodes ?? [];
    const match = (n: typeof nodes[number]) => n.parentId === (props.parentId ?? null) && JSON.stringify(n.plan.source) === JSON.stringify(props.source);
    return nodes.find(n => n.id === workspace?.data.route[props.depth] && match(n)) ?? nodes.filter(match).at(-1);
  });
  const nodeId = useRef(savedNode?.id ?? '');
  const linked = workspace?.data.sharedForks?.some(group => group.nodeIds.includes(nodeId.current));
  const initialChild = workspace?.data.nodes.find(n => savedNode && n.parentId === savedNode.id && n.id === workspace.data.route[props.depth + 1]);
  const [scanning, setScanning] = useState(false), [preview, setPreview] = useState<ForkPlan | null>(null);
  const [targetTokenId, setTargetTokenId] = useState(props.initialTargetTokenId ?? props.source.choice.tokenId);
  const [pair, setPair] = useState<HeadScaleIntervention[] | null>(null);
  const [factor, setFactor] = useState(0), [duration, setDuration] = useState<ForkDuration>('continuation');
  const [limit, setLimit] = useState(Math.min(16, props.manifest.config.contextLength - props.source.prefix.length));
  const [plan, setPlan] = useState<ForkPlan | null>(savedNode?.plan ?? null), [frame, setFrame] = useState<StoryFrame | null>(savedNode?.frame ?? null);
  const [busy, setBusy] = useState(false), [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<StorySelection | null>(initialChild?.plan.source ?? null);
  const draftHeading = useRef<HTMLHeadingElement>(null);
  useEffect(() => { if (preview) { draftHeading.current?.focus({ preventScroll: true }); draftHeading.current?.scrollIntoView({ block: 'start' }); } }, [preview]);
  const panel = useRef<HTMLElement>(null), returnToken = useRef(initialChild && savedNode ? `${initialChild.plan.source.branch}:${initialChild.plan.source.step - savedNode.plan.source.step}` : '');
  const returning = useRef(false);
  const session = useRef<Session | null>(null), worker = useRef<ModelClient | null>(null), loaded = useRef<ModelClient | null>(null);
  const epoch = useRef(0), flight = useRef(false), autoplay = useRef(false);
  const initialized = useRef(false);
  if (!initialized.current && savedNode) { const restored = new Session(savedNode.plan, props.manifest.config); restored.restore(savedNode.frame); session.current = restored; }
  initialized.current = true;
  const encoder = useMemo(() => new BpeTokenizer(props.tokenizer), [props.tokenizer]);
  const common = encoder.preview(props.source.prefix);
  const finished = !!frame?.original.finish && !!frame?.edited?.finish;
  const difference = frame ? firstForkDifference(frame) : null;
  const sharedFrame = linked ? workspace?.data.nodes.find(node => node.id === nodeId.current)?.frame : undefined;
  useEffect(() => {
    if (sharedFrame) { session.current?.restore(sharedFrame); setFrame(sharedFrame); }
  },[sharedFrame]);
  function stop() { ++epoch.current; autoplay.current = false; flight.current = false; worker.current?.dispose(); worker.current = null; setBusy(false); }
  useEffect(() => { if (workspace?.heatmapBusy) stop(); },[workspace?.heatmapBusy]);
  useEffect(() => () => workspace?.setInspectorScanning(false),[workspace?.setInspectorScanning]);
  useEffect(() => () => { ++epoch.current; autoplay.current = false; worker.current?.dispose(); }, []);
  useEffect(() => {
    if (!selection && returning.current) {
      returning.current = false;
      const target = panel.current?.querySelector<HTMLButtonElement>('[data-return-token=true]');
      target?.focus({ preventScroll: true }); target?.scrollIntoView({ block: 'center' });
    }
  }, [selection]);
  function inspect(branch: StorySelection['branch'], index: number) {
    if (!frame || !plan) return;
    stop(); setScanning(false); workspace?.setInspectorScanning(false); props.onStart(); returnToken.current = `${branch}:${index}`;
    setSelection(selectForkToken(plan, frame, branch, index)); props.onInspectChange(true);
  }
  function back() { workspace?.focus(nodeId.current); returning.current = true; setSelection(null); props.onInspectChange(false); }
  function clear() { stop(); session.current = null; setPlan(null); setFrame(null); setError(null); }
  async function run(play: boolean, fresh = false) {
    if (flight.current || props.disabled || scanning || linked && !fresh) return;
    setError(null);
    try {
      if (fresh) {
        if ((workspace?.data.nodes.length ?? 0) >= 500) throw new Error('This workspace has reached 500 comparisons. Download it and reset the story to begin another.');
        const next = new Session({ source: props.source, edit: pair?.[0] ?? { id: props.depth ? `fork-${props.depth}-head` : 'fork-head', kind: 'head.scale', ...props.head, factor, tokens: { kind: 'all' } }, ...(pair ? { additionalEdits: pair.slice(1) } : {}), duration, maxTokens: limit }, props.manifest.config);
        setPreview(null); stop(); nodeId.current = crypto.randomUUID(); session.current = next; setPlan(next.plan); setFrame(next.frame); workspace?.saveNode({ id: nodeId.current, parentId: props.parentId ?? null, plan: next.plan, frame: next.frame });
      }
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); return; }
    const active = session.current;
    if (!active || active.finished) return;
    props.onStart();
    const generation = ++epoch.current, valid = () => epoch.current === generation;
    flight.current = true; autoplay.current = play; setBusy(true);
    const owner = worker.current ?? new ModelClient(); worker.current = owner;
    try {
      if (loaded.current !== owner) { await loadStoryModel(owner, props.manifest); if (!valid()) return; loaded.current = owner; }
      do {
        const next = await active.step((ids, edits) => predictStory(owner, ids, edits));
        if (!valid()) return;
        setFrame(next); workspace?.saveNode({ id: nodeId.current, parentId: props.parentId ?? null, plan: active.plan, frame: next });
        if (!autoplay.current || active.finished) break;
        await new Promise<void>(resolve => setTimeout(resolve, 300));
      } while (valid() && autoplay.current);
    } catch (failure) {
      if (valid()) { setError(failure instanceof Error ? failure.message : String(failure)); owner.dispose(); worker.current = null; }
    } finally { if (valid()) { flight.current = false; autoplay.current = false; setBusy(false); } }
  }
  function story(branch: StoryBranch, name: string, peer: StoryBranch, key: StorySelection['branch']) {
    return <article className="continuation-story" aria-label={`${name} continuation`}>
      <div className="continuation-story-heading"><h3>{name}</h3><span>{branch.finish ? `Stopped · ${branch.finish}` : `${branch.choices.length} new tokens`}</span></div>
      <p className="story-prose"><span>{common}</span><strong>{encoder.preview(branch.tokenIds).slice(common.length)}</strong></p>
      <div className="fork-tokens" aria-label={`${name} continuation tokens`}>{branch.choices.map((choice, i) => <button type="button" key={i} className={choice.tokenId !== peer.choices[i]?.tokenId ? 'fork-different' : ''}
        aria-label={`Inspect ${name.toLowerCase()} token ${props.source.step + i}: ${tokenLabel(props.tokenizer, choice.tokenId)}`}
        data-return-token={returnToken.current === `${key}:${i}`} onClick={() => inspect(key, i)}
        title={`Story step ${props.source.step + i} · token ${choice.tokenId} · model probability ${(choice.probability * 100).toFixed(4)}%`}><small>{props.source.step + i}</small>{tokenLabel(props.tokenizer, choice.tokenId)}</button>)}</div>
    </article>;
  }
  function download() {
    if (!plan || !frame) return;
    const data = { format: 'transformer-microscope-story-fork', schemaVersion: proposedEdits(plan).length === 2 ? 3 : 2, comparisonDepth: props.depth + 1, checkpoint: { modelId: props.manifest.modelId, modelRevision: props.manifest.modelRevision, weightsSha256: props.manifest.weightsSha256, tokenizerSha256: props.manifest.tokenizerSha256 },
      backend: 'cpu', plan, drawOffset: plan.source.step - 1, firstDifferenceStep: difference === null ? null : plan.source.step + difference, frame };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'story-branch-comparison.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  if (selection && plan && frame) return <StoryTokenInspector key={`${selection.branch}:${selection.step}`} selection={selection}
    manifest={props.manifest} tokenizer={props.tokenizer} startStep={plan.source.step} depth={props.depth + 1} parentId={nodeId.current}
    count={frame[selection.branch]!.choices.length} onClose={back} onNavigate={index => inspect(selection.branch, index)} onForkStart={props.onStart} />;
  function chooseScan(edits: HeadScaleIntervention[]) {
    const [edit] = edits; setPair(edits.length === 2 ? structuredClone(edits) : null);
    props.onChooseHead({ layer: edit.layer, head: edit.head }); setFactor(edit.factor);
    setPreview({ source: props.source, edit, ...(edits.length === 2 ? { additionalEdits: edits.slice(1) } : {}), duration, maxTokens: limit });
  }
  return <section ref={panel} className="story-fork" aria-label="Branch from selected token">
    <StoryHeadPairScan key={targetTokenId} targetTokenId={targetTokenId} targeted={props.initialTargetTokenId !== undefined || targetTokenId !== props.source.choice.tokenId} onTargetChange={id => { setTargetTokenId(id); setPreview(null); setPair(null); }} source={props.source} manifest={props.manifest} tokenizer={props.tokenizer} disabled={busy || props.disabled || scanning} onStart={() => { stop(); props.onStart(); }} onChoose={chooseScan} onBusy={value => {setScanning(value);workspace?.setInspectorScanning(value);}} />
    <StoryHeadScan source={props.source} manifest={props.manifest} tokenizer={props.tokenizer} disabled={busy || props.disabled || scanning} onStart={() => { stop(); props.onStart(); }} onChoose={edits => { setTargetTokenId(props.source.choice.tokenId); chooseScan(edits); }} onBusy={value => {setScanning(value);workspace?.setInspectorScanning(value);}} />
    <p className="eyebrow">What if this head worked differently?</p><h3 ref={draftHeading} tabIndex={-1}>Continue two stories from step {props.source.step}.</h3>
    <p>Both paths start with the same text above and regenerate the selected token. Your existing story stays intact. Choose a head in the flow or compare two heads from the scan above.</p>
    {pair ? <div className="head-pair-draft" aria-label="Two-head draft"><strong>{forkEditLabel({ source: props.source, edit: pair[0], additionalEdits: pair.slice(1), duration, maxTokens: limit })}</strong><p>Both changes apply together, with the same duration.</p><button type="button" disabled={busy} onClick={() => { setPair(null); setPreview(null); }}>Use one head instead</button></div> : <div className="fork-head-heading"><strong>Layer {props.head.layer + 1} · Head {props.head.head + 1}</strong><span>Draft change for a new comparison</span></div>}
    <div className="continuation-settings">
      {!pair && <><div className="fork-presets">{[{ value: 0, label: 'Mute' }, { value: 1, label: 'Normal' }, { value: 2, label: 'Amplify' }].map(preset => <button key={preset.value} type="button" disabled={busy} aria-label={`${preset.label} branch head`} aria-pressed={factor === preset.value} onClick={() => setFactor(preset.value)}>{preset.label}</button>)}</div>
      <label>Head strength<input aria-label="Branch head strength" type="number" min="0" max="2" step="0.05" value={factor} disabled={busy} onChange={event => setFactor(Number(event.target.value))} /></label></>}
      <label>Apply edit<select aria-label="Branch edit duration" value={duration} disabled={busy} onChange={event => setDuration(event.target.value as ForkDuration)}><option value="continuation">From this token onward</option><option value="prediction">Only this prediction</option></select></label>
      <label>New tokens<input aria-label="Branch new-token limit" type="number" min="1" max={props.manifest.config.contextLength - props.source.prefix.length} value={limit} disabled={busy} onChange={event => setLimit(Number(event.target.value))} /></label>
    </div>
    <p className="hint">Scales {pair ? 'both heads' : 'this head'} at all input positions {duration === 'prediction' ? 'for the selected prediction, then restores the source settings.' : 'for each prediction from here onward.'} Both paths keep the recorded sampling settings and step-by-step draws.{selectionEdits(props.source).length > 0 && ' The reference keeps the captured edits and their expiry. Your change replaces the same head or adds another head.'}</p>
    <div className="continuation-actions"><button type="button" className="primary" disabled={busy || scanning || props.disabled} onClick={() => void run(true, true)}>{plan ? 'Start new branch comparison' : 'Compare from this token'}</button>
      {plan && <><button type="button" disabled={busy || scanning || finished || props.disabled || linked} onClick={() => void run(false)}>Step comparison</button><button type="button" disabled={busy || scanning || finished || props.disabled || linked} onClick={() => void run(true)}>Play comparison</button><button type="button" disabled={!busy} onClick={stop}>Stop comparison</button><button type="button" onClick={clear}>Clear comparison</button><button type="button" disabled={busy} onClick={download}>Download branch comparison</button></>}
    </div>
    {linked && !finished && <p className="hint">These paths belong to a shared story comparison. Resume both together in Compare interaction maps across story tokens above.</p>}
    {preview && (pair || preview.edit.layer === props.head.layer && preview.edit.head === props.head.head && preview.edit.factor === factor) && <BranchProbabilities targetTokenId={targetTokenId} plan={preview} tokenizer={props.tokenizer} client={props.analysisClient} label="Scanned head probability preview" />}
    {error && <p role="alert" className="error">{error}</p>}
    {plan && frame && <>
      <p className="hint" data-testid="fork-captured-edit">Comparing {forkEditLabel(plan)} · {plan.duration === 'prediction' ? 'selected prediction only' : 'every prediction from this point'}. Source: {plan.source.branch} story, step {plan.source.step}.</p>
      <BranchProbabilities targetTokenId={targetTokenId} plan={plan} tokenizer={props.tokenizer} client={props.analysisClient} />
      <p className="fork-status" role="status">{busy ? 'Comparing · one token at a time…' : finished ? 'Branch comparison complete.' : `Comparison paused after ${frame.step} ${frame.step === 1 ? 'step' : 'steps'}.`}</p>
      <p className="fork-difference" role="status">{difference === null ? frame.step ? 'The generated tokens match so far. A head edit need not change the story.' : 'Waiting for the first pair of tokens.' : `First different token at story step ${plan.source.step + difference}. Highlighted tokens differ at the same generation step.`}</p>
      <div className="continuation-stories paired-stories">{story(frame.original, 'Keep source settings', frame.edited!, 'original')}{story(frame.edited!, 'With head change', frame.original, 'edited')}</div>
    </>}
    <p className="hint">Select a numbered token in either path to inspect it and make another head change. Back to previous comparison returns here with these results intact. The {props.manifest.config.contextLength}-token context limit still applies. Every comparison stays in the story workspace. Use the branch map to reopen it, or download the workspace to keep a copy.</p>
  </section>;
}
