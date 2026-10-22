import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { requireInput, residualHooks, tokenize, textForTokens, tokenLabel, type Intervention, type ModelTokenizer } from './engine/contracts';
import { hookLabel, parseHook, swappedQuery } from './patching';
import type { ComparisonResult, RunSummary } from './engine/runs';
import type { Commands } from './workers/protocol';
import type { ProbeReport } from './engine/probe';
import { createExperiment, parseExperiment, validateExperiment, checkReplay, MAX_EXPERIMENT_BYTES, type ExperimentDocument, type ReplayReport } from './engine/experiment';
import { validateManifest, validateTokenizer } from './engine/weights';
import { StrengthNumber } from './components/StrengthNumber';
import { ComputationInspector } from './components/ComputationInspector';
import { AttentionCellInspector } from './components/AttentionCellInspector';
import { ResidualLens } from './components/ResidualLens';
import { EmbeddingInspector } from './components/EmbeddingInspector';
import { AblationExplorer } from './components/AblationExplorer';
import { InputEditor } from './components/InputEditor';
import { ComputationGraph, stageNames, type FlowNode } from './components/ComputationGraph';
import { NodeInspector } from './components/NodeInspector';
import { PredictionSummary } from './components/PredictionSummary';
import { EffectTrace } from './components/EffectTrace';
import { AttentionRedirect } from './components/AttentionRedirect';
import { ResidualCopy, type CopyDraft } from './components/ResidualCopy';
import { GuidedExperiments, ExperimentGuide } from './components/GuidedExperiments';
import { useEffectTrace } from './hooks/useEffectTrace';
import { matchingStep, traceSteps, type TraceView } from './trace-view';
import { defaultStory, textExamples, fetchTextEvaluation, validateTextEvaluation, type TextEvaluation } from './text-model';
import { VisualFlow } from './components/VisualFlow';
import { StoryContinuation } from './components/StoryContinuation';
import { ModelClient } from './workers/client';
import { fetchExperiments, validateExperiments, retrievalAnswer, type ModelChoice, type ExperimentMetadata, type ExperimentRecipe } from './experiments';

type LoadedModel = Commands['LOAD']['response'];
/** mode 'head' scales a head's output; 'patch' copies a source input's residual row (residual.replace); 'attention' replaces one attention row (attention.override). */
interface Controls { layer: number; head: number; factor: number; scope: string; position: number; target: number; mode: 'head' | 'patch' | 'attention'; hook: string; patchPosition: number; query: number; focus: string }
interface Display { comparison: ComparisonResult; controls: Controls; attention: number[]; raw: number[]; modified: number[]; tokens: number[]; replay?: ReplayReport; sourceTokens?: number[];
  attentionRow?: { baseline: number[]; modified: number[] }; recipe?: ExperimentRecipe }
interface Job { version: number; baseline: RunSummary; controls: Controls; replay?: ExperimentDocument; source?: RunSummary; recipe?: ExperimentRecipe }
const requestedModel = new URLSearchParams(window.location.search).get('model');
const initialChoice: ModelChoice = requestedModel === 'verification' || requestedModel === 'retrieval' ? requestedModel : 'tinystories';
const backendParameter = new URLSearchParams(window.location.search).get('backend');
const initialBackend = backendParameter === 'cpu' || backendParameter === 'webgl' ? backendParameter : 'auto';
const defaults: Controls = { layer: 0, head: 0, factor: .5, scope: 'all', position: 5, target: 2, mode: 'head', hook: '0:block.input', patchPosition: 0, query: 0, focus: 'uniform' };
const shown = (c: Controls) => c.mode === 'patch' ? 'patched residual' : c.mode === 'attention' ? 'redirected attention' : `${c.factor.toFixed(2)}×`;
/** The submitted row: one-hot on a source position, or uniform over every position the query may attend to. */
const focusWeights = (c: Controls, length: number) => Array.from({ length }, (_, i) => c.focus === 'uniform' ? (i <= c.query ? 1 : 0) : (i === Number(c.focus) ? 1 : 0));
/** Screen controls for a saved or prepared intervention; checkSavedIntervention guarantees each kind is representable. */
function restoredControls(base: Controls, patch: Intervention, predictionPosition: number, targetTokenId: number): Controls {
  const common = { ...base, position: predictionPosition, target: targetTokenId };
  if (patch.kind === 'attention.override')
    return { ...common, mode: 'attention', layer: patch.layer, head: patch.head, query: patch.query,
      focus: patch.weights.filter(weight => weight === 1).length === 1 ? String(patch.weights.indexOf(1)) : 'uniform' };
  if (patch.kind === 'residual.replace') return { ...common, mode: 'patch', hook: `${patch.layer}:${patch.stage}`, patchPosition: patch.position };
  return { ...common, mode: 'head', layer: patch.layer, head: patch.head, factor: patch.factor, scope: patch.tokens.kind === 'all' ? 'all' : String(patch.tokens.indices[0]) };
}
const focusLabel = (c: Controls) => c.focus === 'uniform' ? `evenly over positions 0–${c.query}` : `all on position ${c.focus}`;

function SymbolLabel({ tokenizer, tokenId }: { tokenizer: ModelTokenizer; tokenId: number }) {
  return <span className="symbol">{tokenLabel(tokenizer, tokenId)}</span>;
}

export default function App() {
  const [view, setView] = useState<'visual' | 'details'>(() => new URLSearchParams(window.location.search).get('view') === 'details' ? 'details' : 'visual');
  const client = useRef<ModelClient | null>(null);
  const alive = useRef(true);
  const version = useRef(0);
  const baseline = useRef<RunSummary | null>(null);
  const latest = useRef<Job | null>(null);
  const running = useRef<ModelClient | null>(null);
  const modifiedId = useRef<string | null>(null);
  const controlsRef = useRef<Controls>(defaults);
  const baselineInspector = useRef<{ key: string; attention: number[]; raw: number[] } | null>(null);
  const source = useRef<{ owner: ModelClient; run: RunSummary } | null>(null);
  const sourceFlight = useRef<object | null>(null);
  const [choice, setChoice] = useState<ModelChoice>(initialChoice);
  const [textEvaluation, setTextEvaluation] = useState<TextEvaluation | null>(null);
  const [experiments, setExperiments] = useState<ExperimentMetadata | null>(null);
  const [activeRecipe, setActiveRecipe] = useState<ExperimentRecipe | null>(null);
  const [launchingRecipe, setLaunchingRecipe] = useState<string | null>(null);
  const [controls, setControls] = useState(defaults);
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [display, setDisplay] = useState<Display | null>(null);
  const [text, setText] = useState('A E I C G');
  const [tokenIds, setTokenIds] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<ProbeReport | null>(null);
  const [probing, setProbing] = useState(false);
  const [transferring, setTransferring] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{ query: number; source: number } | null>(null);
  const [sessionEnded, setSessionEnded] = useState(false);
  const [sourceText, setSourceText] = useState('');
  const [copying, setCopying] = useState(false);
  const [copyDraft, setCopyDraft] = useState<CopyDraft | null>(null);
  // Inspection is navigation. It does not change the completed intervention until a control is edited.
  const [selectedNode, setSelectedNode] = useState<FlowNode>({ kind: 'head', layer: 0, head: 0 });
  const [traceNavigation, setTraceNavigation] = useState({ key: '', index: 0, origin: 0, position: 0 });

  function createClient(): ModelClient {
    const next = new ModelClient();
    // A crashed worker or lost GPU context can end the session while idle; say so before the next edit.
    next.onEnded = failure => { if (alive.current && client.current === next) { setSessionEnded(true); setError(failure.message); } };
    return next;
  }

  async function drain() {
    const owner = client.current;
    if (!owner || running.current === owner) return;
    running.current = owner;
    try {
      while (latest.current && alive.current && client.current === owner) {
        const job: Job = latest.current;
        latest.current = null;
        const c = job.controls;
        const vectorPosition = c.scope === 'all' ? c.position : Number(c.scope);
        let created: string | undefined;
        try {
          let intervention: Intervention;
          if (c.mode === 'patch') {
            requireInput(job.source, 'Choose a source input to patch from.');
            const hook = parseHook(c.hook);
            const read = await owner.request('INSPECT', { runId: job.source.runId, reference: hook, selection: { position: c.patchPosition } });
            if (job.version !== version.current) continue;
            intervention = { id: 'residual-patch', kind: 'residual.replace', ...hook, position: c.patchPosition, values: Array.from(read.values) };
          } else if (c.mode === 'attention') {
            intervention = { id: 'attention-override', kind: 'attention.override', layer: c.layer, head: c.head, query: c.query, weights: focusWeights(c, job.baseline.tokenIds.length) };
          } else intervention = { id: 'head-strength', kind: 'head.scale', layer: c.layer, head: c.head, factor: c.factor,
            tokens: c.scope === 'all' ? { kind: 'all' } : { kind: 'indices', indices: [Number(c.scope)] } };
          const comparison = await owner.request('COMPARE', {
            baselineRunId: job.baseline.runId, interventions: [intervention], predictionPosition: c.position, targetTokenId: c.target, topK: 8,
          });
          created = comparison.modifiedRunId;
          if (job.version !== version.current) continue;
          // Each mode reads the vectors its panels show; replay checking and display are shared.
          let vectors: { attention: number[]; raw: number[]; modified: number[]; attentionRow?: Display['attentionRow'] };
          if (intervention.kind === 'attention.override') {
            // Overrides change the weights themselves, so this mode shows the modified run's attention.
            const at = { head: c.head, position: c.query };
            const inspectorKey = JSON.stringify([job.baseline.runId, c.layer, c.head, c.query]);
            const savedInspector = baselineInspector.current;
            const readBaseline = async () => {
              if (savedInspector?.key === inspectorKey) return savedInspector;
              const [weights, raw] = await Promise.all([
                owner.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'attention.weights', layer: c.layer }, selection: { head: c.head } }),
                owner.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'attention.head.raw', layer: c.layer }, selection: at }),
              ]);
              return { key: inspectorKey, attention: Array.from(weights.values), raw: Array.from(raw.values) };
            };
            const [weights, changed, inspector] = await Promise.all([
              owner.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'attention.weights', layer: c.layer }, selection: { head: c.head } }),
              owner.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'attention.head.raw', layer: c.layer }, selection: at }),
              readBaseline(),
            ]);
            if (job.version === version.current) baselineInspector.current = inspector;
            const begin = c.query * job.baseline.tokenIds.length, end = begin + job.baseline.tokenIds.length;
            vectors = { attention: Array.from(weights.values), raw: inspector.raw, modified: Array.from(changed.values),
              attentionRow: { baseline: inspector.attention.slice(begin, end), modified: Array.from(weights.values.slice(begin, end)) } };
          } else if (intervention.kind === 'residual.replace') {
            // The patched row: this input's own residual at the patch point, and the source row that replaces it.
            const own = await owner.request('INSPECT', { runId: job.baseline.runId, reference: parseHook(c.hook), selection: { position: c.patchPosition } });
            vectors = { attention: [], raw: Array.from(own.values), modified: intervention.values };
          } else {
            const inspectorKey = JSON.stringify([job.baseline.runId, c.layer, c.head, vectorPosition]);
            const savedInspector = baselineInspector.current;
            const readBaseline = async () => {
              if (savedInspector?.key === inspectorKey) return savedInspector;
              const [attention, raw] = await Promise.all([
                owner.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'attention.weights', layer: c.layer }, selection: { head: c.head } }),
                owner.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'attention.head.raw', layer: c.layer }, selection: { head: c.head, position: vectorPosition } }),
              ]);
              return { key: inspectorKey, attention: Array.from(attention.values), raw: Array.from(raw.values) };
            };
            const [inspector, modified] = await Promise.all([
              readBaseline(),
              owner.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'attention.head.modified', layer: c.layer }, selection: { head: c.head, position: vectorPosition } }),
            ]);
            if (job.version === version.current) baselineInspector.current = inspector;
            vectors = { attention: inspector.attention, raw: inspector.raw, modified: Array.from(modified.values) };
          }
          if (job.version !== version.current) continue;
          let replay: ReplayReport | undefined;
          if (job.replay) {
            const [baseLogits, modifiedLogits] = await Promise.all([
              owner.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'logits' }, selection: { position: c.position } }),
              owner.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'logits' }, selection: { position: c.position } }),
            ]);
            if (job.version !== version.current) continue;
            replay = checkReplay(job.replay, baseLogits.values, modifiedLogits.values, comparison.execution.backend);
          }
          if (modifiedId.current) void owner.request('RELEASE', { runId: modifiedId.current }).catch(() => {});
          modifiedId.current = created; created = undefined;
          setDisplay({ comparison, controls: c, tokens: job.baseline.tokenIds, ...vectors, replay, sourceTokens: job.source?.tokenIds, recipe: job.recipe });
          setError(null);
        } catch (failure) {
          if (alive.current && job.version === version.current) setError(failure instanceof Error ? failure.message : String(failure));
        } finally {
          if (created) void owner.request('RELEASE', { runId: created }).catch(() => {});
        }
      }
    } finally { if (running.current === owner) { running.current = null; if (alive.current) setPending(false); } }
  }

  function schedule(next: Controls, run = baseline.current, recipe: ExperimentRecipe | null = null, replay?: ExperimentDocument) {
    if (next.mode === 'patch') setCopyDraft({ hook: next.hook, position: next.patchPosition });
    else { setCopyDraft(null); sourceFlight.current = null; setCopying(false); }
    setSelectedNode(next.mode === 'patch' ? { kind: 'stage', ...parseHook(next.hook) } : { kind: 'head', layer: next.layer, head: next.head });
    setActiveRecipe(recipe);
    controlsRef.current = next; setControls(next);
    const current = ++version.current;
    if (!run) return;
    latest.current = { version: current, baseline: run, controls: { ...next }, replay, source: next.mode === 'patch' ? source.current?.run : undefined, recipe: recipe ?? undefined };
    setPending(true); void drain();
  }

  /** A source run belongs to one target input and worker; it is released whenever either changes. */
  function releaseSource(owner = client.current) {
    sourceFlight.current = null; setCopying(false);
    const old = source.current;
    source.current = null;
    if (old && old.owner === owner) void owner.request('RELEASE', { runId: old.run.runId }).catch(() => {});
  }

  async function applySource(text: string, restored?: Controls, replay?: ExperimentDocument, loaded = model) {
    const target = baseline.current, owner = client.current;
    if (!loaded || !target || !owner) return;
    const flight = {}, generation = version.current;
    sourceFlight.current = flight; setCopying(true);
    setError(null); setSourceText(text);
    try {
      const tokens = tokenize(text, loaded.tokenizer, loaded.manifest.config.contextLength);
      requireInput(tokens.length === target.tokenIds.length, `The source must have the same number of symbols as the input (${target.tokenIds.length - 1}).`);
      const reusable = source.current?.owner === owner && JSON.stringify(source.current.run.tokenIds) === JSON.stringify(tokens) ? source.current.run : undefined;
      const run = reusable ?? await owner.request('BASELINE', { tokenIds: tokens });
      if (!alive.current || client.current !== owner || baseline.current !== target || sourceFlight.current !== flight || generation !== version.current) {
        if (!reusable) void owner.request('RELEASE', { runId: run.runId }).catch(() => {});
        return;
      }
      if (!reusable) releaseSource(owner);
      source.current = { owner, run };
      const differing = tokens.findIndex((token, i) => token !== target.tokenIds[i]);
      schedule(restored ? { ...restored, mode: 'patch' }
        : { ...controlsRef.current, mode: 'patch', patchPosition: differing >= 0 ? differing : Math.min(controlsRef.current.patchPosition, tokens.length - 1) }, target, null, replay);
    } catch (failure) { if (alive.current && client.current === owner && sourceFlight.current === flight) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (sourceFlight.current === flight) { sourceFlight.current = null; if (alive.current) setCopying(false); } }
  }

  function chooseMode(mode: Controls['mode'], base = controlsRef.current) {
    if (mode === 'head') { schedule({ ...base, mode }); return; }
    if (mode === 'attention') { schedule({ ...base, mode, query: base.position, focus: 'uniform' }); return; }
    const target = baseline.current, loaded = model;
    if (!target || !loaded) return;
    const own = textForTokens(target.tokenIds, loaded.tokenizer);
    // Text left from another input or checkpoint may no longer fit; then start from a suggestion that does.
    const fits = (text: string) => { try { return tokenize(text, loaded.tokenizer, loaded.manifest.config.contextLength).length === target.tokenIds.length; } catch { return false; } };
    const typed = loaded.tokenizer.kind === 'byte-bpe' ? sourceText : sourceText.trim();
    const proposed = typed && fits(typed) ? typed : swappedQuery(target.tokenIds, loaded.tokenizer) ?? own;
    const preview = tokenize(proposed, loaded.tokenizer, loaded.manifest.config.contextLength);
    const differing = preview.findIndex((token, i) => token !== target.tokenIds[i]);
    const position = copyDraft?.position ?? (controlsRef.current.mode === 'patch' ? controlsRef.current.patchPosition : differing >= 0 ? differing : 0);
    setSourceText(proposed); draftCopy({ hook: base.hook, position });
    requestAnimationFrame(() => document.querySelector('.residual-copy')?.scrollIntoView({ block: 'start', behavior: 'instant' }));
  }

  async function establish(loaded: LoadedModel, input: string, generation: number, recipe: ExperimentRecipe | null = null, replay?: ExperimentDocument, settings?: Controls) {
    const owner = client.current!;
    const tokens = tokenize(input, loaded.tokenizer, loaded.manifest.config.contextLength);
    const next = await owner.request('BASELINE', { tokenIds: tokens });
    if (!alive.current || generation !== version.current || client.current !== owner) {
      void owner.request('RELEASE', { runId: next.runId }).catch(() => {}); return;
    }
    let textTarget: number | undefined;
    if (loaded.tokenizer.kind === 'byte-bpe' && !recipe && !replay && !settings) {
      const logits = await owner.request('INSPECT', { runId: next.runId, reference: { stage: 'logits' }, selection: { position: tokens.length - 1 } });
      textTarget = logits.values.reduce((best, value, index, values) => value > values[best] ? index : best, 0);
      if (!alive.current || generation !== version.current || client.current !== owner) { void owner.request('RELEASE', { runId: next.runId }).catch(() => {}); return; }
    }
    const old = baseline.current;
    baseline.current = next; setTokenIds(tokens); setSelectedCell(null);
    if (old) void owner.request('RELEASE', { runId: old.runId }).catch(() => {});
    if (modifiedId.current) {
      void owner.request('RELEASE', { runId: modifiedId.current }).catch(() => {}); modifiedId.current = null;
    }
    releaseSource(owner);
    const restored = replay ? restoredControls(controlsRef.current, replay.observation.interventions[0], replay.observation.predictionPosition, replay.observation.targetTokenId)
      : recipe ? restoredControls(controlsRef.current, recipe.intervention, recipe.predictionPosition, recipe.targetTokenId) : undefined;
    if (replay && restored?.mode === 'patch') {
      // A saved patch needs its source run first; show the baseline until the replayed patch completes.
      schedule({ ...restored, mode: 'head', factor: 1 }, next);
      await applySource(replay.observation.source!.text, restored, replay, loaded);
      return;
    }
    schedule(settings ?? restored ?? { ...controlsRef.current, mode: 'head', position: tokens.length - 1, scope: 'all', target: textTarget ?? retrievalAnswer(tokens, loaded.tokenizer) ?? controlsRef.current.target },
      next, recipe, replay);
  }

  async function load(backend: 'auto' | 'cpu' | 'webgl', selected: ModelChoice, input?: string, restored?: ExperimentDocument, settings?: Controls) {
    const owner = client.current;
    const generation = ++version.current;
    latest.current = null; baseline.current = null; modifiedId.current = null; baselineInspector.current = null; source.current = null;
    sourceFlight.current = null; setCopying(false); setCopyDraft(null);
    setChoice(selected); setTextEvaluation(null); setExperiments(null); setActiveRecipe(null); setLaunchingRecipe(null);
    setLoading(true); setModel(null); setDisplay(null); setError(null); setProbe(null);
    try {
      const manifestUrl = new URL(`${import.meta.env.BASE_URL}models/${selected}/manifest.json`, window.location.origin).href;
      const [loaded, data] = await Promise.all([
        owner!.request('LOAD', { manifestUrl, backend }),
        selected === 'retrieval' ? fetchExperiments(manifestUrl) : selected === 'tinystories' ? fetchTextEvaluation(manifestUrl) : Promise.resolve(null),
      ]);
      if (!alive.current || generation !== version.current) return;
      if (restored) validateExperiment(restored, loaded.manifest, loaded.tokenizer);
      const metadata = selected === 'retrieval' && data ? validateExperiments(data as unknown[], loaded.manifest, loaded.tokenizer) : null;
      if (selected === 'tinystories') setTextEvaluation(validateTextEvaluation(data, loaded.manifest.modelRevision));
      const recipe = input === undefined ? metadata?.recipes[0] ?? null : null;
      const sequence = input ?? recipe?.text ?? (selected === 'tinystories' ? defaultStory : 'A E I C G');
      if (input === undefined) controlsRef.current = selected === 'tinystories' ? { ...defaults, factor: 1 } : defaults;
      setExperiments(metadata); setModel(loaded); setText(sequence);
      await establish(loaded, sequence, generation, recipe, restored, settings);
    } catch (failure) {
      if (alive.current && generation === version.current) setError(failure instanceof Error ? failure.message : String(failure));
    } finally { if (alive.current && client.current === owner) setLoading(false); }
  }

  async function selectRecipe(id: string) {
    const recipe = experiments?.recipes.find(item => item.id === id);
    if (!recipe || !model) return;
    const owner = client.current;
    const generation = ++version.current;
    latest.current = null; setLaunchingRecipe(id); setLoading(true); setError(null); setText(recipe.text);
    try { await establish(model, recipe.text, generation, recipe); }
    catch (failure) { if (alive.current && client.current === owner) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (alive.current && client.current === owner) { setLoading(false); setLaunchingRecipe(null); } }
  }

  useEffect(() => {
    alive.current = true;
    client.current = createClient();
    void load(initialBackend, initialChoice);
    return () => { alive.current = false; ++version.current; latest.current = null; client.current?.dispose(); };
  }, []);

  async function runInput(event: FormEvent) {
    event.preventDefault();
    if (!model) return;
    const owner = client.current;
    const generation = ++version.current;
    latest.current = null; setLoading(true); setError(null);
    try { await establish(model, text, generation); }
    catch (failure) { if (alive.current && client.current === owner) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (alive.current && client.current === owner) setLoading(false); }
  }

  async function restart() {
    const sequence = display && model ? textForTokens(display.tokens, model.tokenizer) : undefined;
    // Source runs belong to the old worker; the patch is restored by re-running the active source (not unsubmitted edits) in the new one.
    const active = source.current;
    const patchSource = controlsRef.current.mode === 'patch' && active && model ? textForTokens(active.run.tokenIds, model.tokenizer) : null;
    const settings = display ? { ...controlsRef.current, ...(patchSource !== null ? { mode: 'head' as const } : {}) } : undefined;
    ++version.current; latest.current = null;
    client.current?.dispose(); client.current = createClient(); setProbing(false); setSessionEnded(false);
    await load('cpu', choice, sequence, undefined, settings);
    if (patchSource !== null && alive.current && baseline.current) await applySource(patchSource, settings);
  }

  async function verify() {
    const owner = client.current;
    if (!owner) return;
    setProbing(true); setError(null); setProbe(null);
    try { const report = await owner.request('PROBE', {}); if (alive.current && client.current === owner) setProbe(report); }
    catch (failure) { if (alive.current && client.current === owner) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (alive.current && client.current === owner) setProbing(false); }
  }

  async function exportExperiment() {
    const completed = display, loaded = model, owner = client.current;
    if (!completed || !loaded || !owner) return;
    setTransferring(true); setError(null);
    try {
      const [original, modified] = await Promise.all([
        owner.request('INSPECT', { runId: completed.comparison.baselineRunId, reference: { stage: 'logits' }, selection: { position: completed.controls.position } }),
        owner.request('INSPECT', { runId: completed.comparison.modifiedRunId, reference: { stage: 'logits' }, selection: { position: completed.controls.position } }),
      ]);
      const document = createExperiment(loaded.manifest, loaded.tokenizer, completed.tokens, {
        predictionPosition: completed.controls.position, targetTokenId: completed.controls.target, interventions: completed.comparison.interventions,
        sourceTokenIds: completed.controls.mode === 'patch' ? completed.sourceTokens : undefined,
      }, original.values, modified.values, { backend: loaded.backend, precision: loaded.precision });
      const url = URL.createObjectURL(new Blob([JSON.stringify(document, null, 2) + '\n'], { type: 'application/json' }));
      const link = window.document.createElement('a');
      link.href = url; link.download = `${loaded.manifest.modelId}-experiment.json`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (failure) { if (alive.current) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (alive.current) setTransferring(false); }
  }

  async function importExperiment(file: File) {
    if (!model) return;
    setTransferring(true); setError(null);
    try {
      if (file.size > MAX_EXPERIMENT_BYTES) throw new Error('Experiment files must be at most 1 MiB.');
      const document = parseExperiment(await file.text());
      const selected: ModelChoice | undefined = document.checkpoint.modelId === 'retrieval-seed-2026' ? 'retrieval'
        : document.checkpoint.modelId === 'verification-seed-1729' ? 'verification' : document.checkpoint.modelId === 'text-tinystories-seed-2026' ? 'tinystories' : undefined;
      if (!selected) throw new Error('This saved model is not installed. Choose an experiment exported by an available checkpoint.');
      // Validate before replacing the visible experiment or loading another GPU model.
      if (selected === choice) validateExperiment(document, model.manifest, model.tokenizer);
      else {
        const directory = new URL(`${import.meta.env.BASE_URL}models/${selected}/`, window.location.origin);
        const [rawManifest, rawTokenizer] = await Promise.all(['manifest.json', 'tokenizer.json'].map(async name => {
          const response = await fetch(new URL(name, directory));
          if (!response.ok) throw new Error('Could not load the saved checkpoint. Try again.');
          return response.json();
        }));
        const manifest = validateManifest(rawManifest);
        validateExperiment(document, manifest, validateTokenizer(rawTokenizer, manifest.config.vocabularySize));
      }
      if (!alive.current) return;
      if (selected !== choice) await load('auto', selected, document.input.text, document);
      else {
        const generation = ++version.current;
        latest.current = null; setLoading(true); setText(document.input.text);
        try { await establish(model, document.input.text, generation, null, document); }
        finally { if (alive.current) setLoading(false); }
      }
    } catch (failure) { if (alive.current) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (alive.current) setTransferring(false); }
  }

  const tokenizer = model?.tokenizer;
  const c = model?.manifest.config;
  const symbols = useMemo(() => tokenizer?.tokens.map((_, id) => tokenLabel(tokenizer, id)), [tokenizer]);
  const rows = display ? display.comparison.displayTokens : [];
  const length = display?.tokens.length ?? 0;

  const locked = loading || probing || transferring || copying || !model || sessionEnded;
  const tracing = useEffectTrace(client.current, display?.comparison ?? null, pending || locked);
  const report = tracing.report;
  const origin = report?.intervention.kind === 'attention.override' ? report.intervention.query
    : report?.intervention.kind === 'residual.replace' ? report.intervention.position
    : report?.intervention.tokens.kind === 'indices' ? report.intervention.tokens.indices[0] ?? report.predictionPosition : report?.predictionPosition ?? 0;
  const navigation = report && traceNavigation.key === report.modifiedRunId ? traceNavigation : { key: report?.modifiedRunId ?? '', index: 0, origin, position: origin };
  const steps = useMemo(() => report ? traceSteps(report, navigation.origin) : [], [report, navigation.origin]);
  const measurements = useMemo(() => new Map(report?.nodes.map(node => [node.key, node]) ?? []), [report]);
  const traceView: TraceView | undefined = useMemo(() => report ? { report, position: navigation.position, origin: navigation.origin, measurements } : undefined, [report, navigation.position, navigation.origin, measurements]);
  useEffect(() => {
    if (!report || traceNavigation.key === report.modifiedRunId) return;
    setTraceNavigation({ key: report.modifiedRunId, index: 0, origin, position: origin });
    setSelectedNode(traceSteps(report, origin)[0].node);
  }, [report?.modifiedRunId, traceNavigation.key]);
  const headControls: Controls = selectedNode.kind === 'head' && (selectedNode.layer !== controls.layer || selectedNode.head !== controls.head || controls.mode === 'patch')
    ? { ...controls, mode: 'head', layer: selectedNode.layer, head: selectedNode.head, factor: 1, scope: 'all' } : controls;
  const patchNode = controls.mode === 'patch' && selectedNode.kind === 'stage' && selectedNode.layer === parseHook(controls.hook).layer && selectedNode.stage === parseHook(controls.hook).stage;
  const residualNode = selectedNode.kind === 'stage' && residualHooks.some(hook => hook === selectedNode.stage);
  const sourcePreview = useMemo(() => {
    if (!copyDraft || !tokenizer || !c) return {};
    try {
      const tokens = tokenize(sourceText, tokenizer, c.contextLength);
      requireInput(tokens.length === tokenIds.length, `The source must have the same number of symbols as the input (${tokenIds.length - 1}).`);
      return { tokens };
    } catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [!!copyDraft, sourceText, tokenizer, c, tokenIds]);
  const activeHead = display && selectedNode.kind === 'head' && display.controls.mode !== 'patch' && selectedNode.layer === display.controls.layer && selectedNode.head === display.controls.head;
  const inspectPosition = traceView ? selectedNode.kind === 'readout' ? traceView.report.predictionPosition : traceView.position
    : copyDraft && selectedNode.kind === 'stage' && `${selectedNode.layer}:${selectedNode.stage}` === copyDraft.hook ? copyDraft.position : display ? activeHead ? display.controls.mode === 'attention' ? display.controls.query : display.controls.scope === 'all' ? display.controls.position : Number(display.controls.scope)
    : patchNode ? display.controls.patchPosition : display.controls.position : 0;
  const selectionTitle = selectedNode.kind === 'head' ? `Layer ${selectedNode.layer + 1} · Head ${selectedNode.head + 1}${selectedNode.part === 'attention' ? ' attention weights' : ''}`
    : selectedNode.kind === 'stage' ? `Layer ${selectedNode.layer + 1} · ${stageNames[selectedNode.stage]}` : selectedNode.kind === 'embedding' ? 'Input embeddings' : 'Final readout';
  const completedText = display && tokenizer ? textForTokens(display.tokens, tokenizer) : '';
  const canonicalDraft = (draft: string) => tokenizer?.kind === 'byte-bpe' ? draft : draft.trim().split(/\s+/u).join(' ');
  const inputDraft = !!display && canonicalDraft(text) !== completedText;
  const completedSource = display?.sourceTokens && tokenizer ? textForTokens(display.sourceTokens, tokenizer) : '';
  const hasDraft = inputDraft || !!copyDraft && (display?.controls.mode !== 'patch' || copyDraft.hook !== display.controls.hook || copyDraft.position !== display.controls.patchPosition || canonicalDraft(sourceText) !== completedSource);
  const completedRecipe = display?.comparison.execution.runtimeGeneration === model?.runtimeGeneration ? display?.recipe : undefined;
  function inspectNode(node: FlowNode) {
    setSelectedNode(node);
    if (copyDraft && node.kind === 'stage' && residualHooks.some(hook => hook === node.stage)) setCopyDraft({ ...copyDraft, hook: `${node.layer}:${node.stage}` });
    if (traceView) {
      const position = node.kind === 'readout' ? report!.predictionPosition : navigation.position;
      setTraceNavigation({ ...navigation, position, index: matchingStep(steps, node, position) });
    }
    if (view === 'details' && window.matchMedia('(max-width: 800px)').matches) requestAnimationFrame(() => {
      document.querySelector('.node-panel')?.scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    });
  }
  function moveTrace(index: number) {
    const step = steps[index];
    if (!step) return;
    setTraceNavigation({ ...navigation, index, position: step.position }); setSelectedNode(step.node);
    requestAnimationFrame(() => document.querySelector('.computation-graph .flow-node[aria-pressed=true]')?.scrollIntoView({ block: 'center', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' }));
  }
  function redirectQuery(query: number) {
    const focus = headControls.focus;
    schedule({ ...headControls, query, focus: focus !== 'uniform' && Number(focus) > query ? String(query) : focus });
  }
  function draftCopy(draft: CopyDraft) {
    const node: FlowNode = { kind: 'stage', ...parseHook(draft.hook) };
    setSelectedNode(node);
    setCopyDraft(draft);
    if (traceView) setTraceNavigation({ ...navigation, position: draft.position, index: matchingStep(steps, node, draft.position) });
  }
  function openCopy() {
    setView('details');
    if (selectedNode.kind !== 'stage' || !residualNode) return;
    chooseMode('patch', { ...controlsRef.current, hook: `${selectedNode.layer}:${selectedNode.stage}` });
  }
  function traceExample() {
    setView('details');
    if (report) { moveTrace(0); return; }
    if (tracing.error) tracing.retry();
    setTraceNavigation({ key: '', index: 0, origin: 0, position: 0 }); tracing.setOpen(true);
  }
  function inspectEditedHead() {
    const initiatingElement = document.activeElement;
    setView('details');
    const patch = display?.comparison.interventions[0];
    if (!patch || patch.kind !== 'head.scale') return;
    const node: FlowNode = { kind: 'head', layer: patch.layer, head: patch.head };
    inspectNode(node);
    const position = patch.tokens.kind === 'indices' ? patch.tokens.indices[0] : display!.comparison.predictionPosition;
    if (traceView) setTraceNavigation({ ...navigation, position, index: matchingStep(steps, node, position) });
    requestAnimationFrame(() => {
      // Navigation must not take focus back after the user has moved to another control.
      if (document.activeElement !== initiatingElement) return;
      const selected = document.querySelector<HTMLButtonElement>('.computation-graph .flow-node[aria-pressed=true]');
      selected?.focus({ preventScroll: true });
      if (!window.matchMedia('(max-width: 800px)').matches) selected?.scrollIntoView({ block: 'center', behavior: 'instant' });
    });
  }

  return <main className={`flow-app view-${view} ${tracing.open ? 'trace-active' : ''}`}>
    <header className="masthead">
      <div className="wordmark"><span className="mark" aria-hidden="true">μ</span><div><p className="eyebrow">Follow a prediction through the model</p><h1>Transformer Microscope</h1></div></div>
      <span className={`status ${pending || loading || copying ? 'working' : ''}`} role="status">{loading ? 'Loading model…' : copying ? 'Preparing source input…' : pending ? `Computing…${display ? ` previous ${shown(display.controls)}` : ''}` : display ? `Showing ${shown(display.controls)}` : 'Ready'}</span>
    </header>
    <div className="view-toolbar"><p>A transformer learns how symbols relate. Explore how one small change affects its answer.</p>
      <div role="group" aria-label="Workspace view"><button type="button" aria-pressed={view === 'visual'} onClick={() => setView('visual')}>Visual flow</button><button type="button" aria-pressed={view === 'details'} onClick={() => setView('details')}>Detailed</button></div></div>
    <div className="notice"><strong>{choice === 'tinystories' ? 'TinyStories · real text' : choice === 'retrieval' ? 'A simple memory game' : 'Verification model'}</strong>
      <span>{choice === 'tinystories' ? 'Enter a short passage and see which word or word-piece comes next. This experimental model learned from children’s stories.' : choice === 'retrieval' ? 'Give each letter a number, then ask the model to recall one. This small trained model helps us see how a transformer works.' : 'Random weights, real computations. These predictions verify the engine; they do not demonstrate learned language.'}</span></div>
    <section className="input-workbench" aria-label="Input editor">
      <div className="setup-row">
        <div className="step-heading"><span className="step-number">1</span><div><p className="eyebrow">Set the input</p><h2>{choice === 'tinystories' ? 'Start a short story' : 'Give the model a question'}</h2></div></div>
        <label>Model<select aria-label="Model" disabled={loading || probing || transferring} value={choice} onChange={event => void load('auto', event.target.value as ModelChoice)}>
          <option value="tinystories">TinyStories · real text (experimental)</option><option value="retrieval">Trained retrieval model</option><option value="verification">Untrained verification model</option></select></label>
        {experiments && <label>Prepared experiment<select aria-label="Prepared experiment" disabled={locked} value={activeRecipe?.id ?? ''} onChange={event => void selectRecipe(event.target.value)}>
          <option value="" disabled>Custom experiment</option>{experiments.recipes.map(recipe => <option key={recipe.id} value={recipe.id}>{recipe.title}</option>)}</select></label>}
      </div>
      {experiments && <details className="guided-choices" open={view === 'details' ? true : undefined}><summary>Try a guided experiment</summary><GuidedExperiments recipes={experiments.recipes} completed={completedRecipe?.id} launching={launchingRecipe}
        disabled={pending || locked} draft={hasDraft} onLaunch={id => void selectRecipe(id)} /></details>}
      <InputEditor key={model?.manifest.modelId ?? 'loading'} choice={choice} text={text} disabled={locked} busy={loading}
        draft={inputDraft} tokenizer={tokenizer} contextLength={c?.contextLength} onChange={setText} onSubmit={runInput} />
      {choice === 'tinystories' && <div className="story-examples" role="group" aria-label="Story examples"><span>Try a prompt</span>{textExamples.map(example => <button type="button" key={example.label} disabled={locked} onClick={() => setText(example.text)}>{example.label}</button>)}</div>}
      {choice === 'tinystories' && display && <a className="continuation-jump" href="#story-continuation">Watch a story grow token by token ↓</a>}
      {textEvaluation && <details className="text-model-about"><summary>About this trained model</summary><p>Trained locally on TinyStories V2 · {textEvaluation.training.step.toLocaleString()} steps · {c?.layers} layers · {c?.heads} heads per layer. Validation loss: {textEvaluation.training.validation.mean.toFixed(3)} nats/token; bigram baseline: {textEvaluation.baselines.bigram.toFixed(3)}.</p><p>Experimental: the original name-copying gate failed ({(textEvaluation.nameCopying.rate * 100).toFixed(1)}%, required 80%). Pronoun agreement: {(textEvaluation.pronounAgreement.rate * 100).toFixed(0)}%. These template scores describe measured tasks, not general language ability. Full-run determinism and the untouched audit set remain unevaluated.</p><a href="https://huggingface.co/datasets/roneneldan/TinyStories" target="_blank" rel="noreferrer">Dataset and source</a></details>}
    </section>
    {error && <p className="error" role="alert">{error}
      {sessionEnded && <button type="button" className="secondary recover" onClick={() => void restart()}>Restart on CPU</button>}</p>}
    {view === 'visual' && c && display && tokenizer && <VisualFlow config={c} tokens={display.tokens.map(id => symbols![id])} selected={selectedNode}
      comparison={display.comparison} attention={display.attention} layer={display.controls.layer} head={display.controls.head}
      position={display.controls.mode === 'attention' ? display.controls.query : display.controls.position} pending={pending || loading || copying} ended={sessionEnded}
      disabled={locked} onSelect={inspectNode} onDetails={() => setView('details')}
      headControl={headControls.mode === 'head' ? { factor: headControls.factor, scope: headControls.scope === 'all' ? 'Applies at every input position' : `Applies at position ${headControls.scope}`,
        onChange: factor => schedule({ ...headControls, factor }) } : undefined}
      onReset={() => schedule({ ...controlsRef.current, mode: 'head', factor: 1 })} />}
    {choice === 'tinystories' && model && display && tokenizer?.kind === 'byte-bpe' && <StoryContinuation
      key={model.runtimeGeneration} inputKey={display.comparison.baselineRunId} manifest={model.manifest} tokenizer={tokenizer}
      tokens={display.tokens} prompt={completedText} draft={inputDraft} disabled={loading || pending || probing || sessionEnded}
      edit={display.comparison.interventions[0]?.kind === 'head.scale' && display.comparison.interventions[0].tokens.kind === 'all' && display.comparison.interventions[0].factor !== 1 ? display.comparison.interventions[0] : undefined} />}
    <div className="workspace flow-workspace">
      <div className="exploration">
        <section className="graph-pane" aria-label="Computation workspace" hidden={view === 'visual'}>
          <div className="step-heading"><span className="step-number">2</span><div><p className="eyebrow">Explore the computation</p><h2>Choose where to look</h2></div></div>
          <p className="hint">Click a node to inspect the completed run. {c?.heads ?? 4} heads in each layer run in parallel; arrows show computation order.</p>
          {display && <ExperimentGuide recipe={completedRecipe} comparison={display.comparison} target={display.controls.target}
            disabled={pending || locked} pending={pending || loading || copying} ended={sessionEnded} draft={hasDraft} traceOpen={!!report}
            onInspect={inspectEditedHead} onTrace={traceExample} />}
          {copyDraft && tokenizer && c && <ResidualCopy draft={copyDraft} sourceText={sourceText} currentText={completedText} tokens={tokenIds} sourceTokens={sourcePreview.tokens} symbols={symbols!} layers={c.layers}
            disabled={locked} pending={pending || copying} validation={sourcePreview.error} alternate={swappedQuery(tokenIds, tokenizer)}
            onSource={setSourceText} onDraft={draftCopy} onCopy={() => void applySource(sourceText, { ...controlsRef.current, mode: 'patch', hook: copyDraft.hook, patchPosition: copyDraft.position })}
            onClose={() => setCopyDraft(null)} completed={display?.comparison.interventions[0]?.kind === 'residual.replace' && display.sourceTokens
              ? { patch: display.comparison.interventions[0], sourceTokens: display.sourceTokens, baseline: display.raw, copied: display.modified } : undefined} />}
          <EffectTrace open={tracing.open} supported={tracing.supported} busy={pending || locked} ended={sessionEnded} report={report} error={tracing.error}
            position={navigation.position} origin={navigation.origin} steps={steps} index={navigation.index} symbols={symbols ?? []}
            onOpen={open => { if (open) setTraceNavigation({ key: '', index: 0, origin: 0, position: 0 }); tracing.setOpen(open); }} onRetry={tracing.retry} onMove={moveTrace}
            onPosition={position => { setTraceNavigation({ ...navigation, origin: position, position, index: 0 }); if (report) setSelectedNode(traceSteps(report, position)[0].node); }} />
          <div className="prediction-position"><span>Predict after</span><div className="tokens" aria-label="Input tokens">{tokenizer && tokenIds.map((id, index) => <button key={index} disabled={locked}
            className={controls.position === index ? 'token selected' : 'token'} onClick={() => schedule({ ...controlsRef.current, position: index })}
            aria-label={`Predict after position ${index}: ${symbols![id]}`} aria-pressed={controls.position === index}>
            <small>{index}</small>{symbols![id]}</button>)}</div></div>
          {c ? <ComputationGraph key={model!.runtimeGeneration} config={c} selected={selectedNode} intervention={display?.comparison.interventions[0]} disabled={locked || !display} onSelect={inspectNode}
            trace={traceView} symbols={symbols} target={display?.controls.target} sourceTokens={display?.sourceTokens} />
            : <p className="hint">Loading the model's computation graph…</p>}
        </section>
        <aside className="controls-panel node-panel" aria-label="Selected node">
          <div className="node-panel-heading"><p className="eyebrow">Selected node</p><h2>{selectionTitle}</h2><span className="inspection-note">Inspection keeps the active experiment intact</span></div>
          {view === 'details' && display && tokenizer && client.current && !loading && display.comparison.baselineRunId === baseline.current?.runId && <NodeInspector node={selectedNode} client={client.current}
            baselineRunId={display.comparison.baselineRunId} modifiedRunId={display.comparison.modifiedRunId} position={inspectPosition}
            symbol={symbols![display.tokens[inspectPosition]]} trace={traceView} target={display.controls.target} symbols={symbols!} sourceTokens={display.sourceTokens}
            preset={patchNode && selectedNode.kind === 'stage' && display.comparison.interventions[0]?.kind === 'residual.replace' && selectedNode.layer === display.comparison.interventions[0].layer && selectedNode.stage === display.comparison.interventions[0].stage && inspectPosition === display.controls.patchPosition
              || activeHead && !(selectedNode.kind === 'head' && selectedNode.part === 'attention') && inspectPosition === (display.controls.mode === 'attention' ? display.controls.query : display.controls.scope === 'all' ? display.controls.position : Number(display.controls.scope)) ? { baseline: display.raw, modified: display.modified } : undefined} />}
          {selectedNode.kind === 'head' || residualNode ? <>
        <section>
          <p className="eyebrow">At the selected node</p><h3>{residualNode ? 'Copy at this residual junction' : headControls.mode === 'attention' ? "Redirect a head's attention" : "Change a head's contribution"}</h3>
          <label>Intervention type<select aria-label="Intervention type" disabled={locked} value={residualNode ? 'patch' : headControls.mode} onChange={event => { if (event.target.value === 'patch') setView('details'); chooseMode(event.target.value as Controls['mode'], headControls); }}>
            <option value="head">Scale a head's output</option><option value="attention">Redirect a head's attention</option><option value="patch">Patch the residual from another input</option></select></label>
          {residualNode ? <div className="patch-controls">
            <p className="hint">Copies the source input's vector into {selectedNode.kind === 'stage' ? hookLabel(`${selectedNode.layer}:${selectedNode.stage}`) : 'this junction'}. Source and position choices stay a draft until you press Copy from source.</p>
            <button type="button" className="secondary" disabled={locked} onClick={openCopy}>Choose source and position</button>
          </div> : <>
          <div className="paired"><label>Layer<select aria-label="Layer" disabled={loading || probing || transferring || !c} value={headControls.layer} onChange={event => schedule({ ...headControls, layer: Number(event.target.value) })}>
            {c && Array.from({ length: c.layers }, (_, i) => <option value={i} key={i}>Layer {i + 1}</option>)}</select></label>
            <label>Head<select aria-label="Head" disabled={loading || probing || transferring || !c} value={headControls.head} onChange={event => schedule({ ...headControls, head: Number(event.target.value) })}>
              {c && Array.from({ length: c.heads }, (_, i) => <option value={i} key={i}>Head {i + 1}</option>)}</select></label></div>
          {headControls.mode === 'attention' ? <div className="attention-controls">
            {tokenizer && <AttentionRedirect tokens={tokenIds} symbols={symbols!} query={headControls.query} focus={headControls.focus} disabled={locked} pending={pending}
              showWeights={!(traceView && selectedNode.kind === 'head' && selectedNode.part === 'attention' && traceView.position === display?.controls.query)}
              onQuery={redirectQuery} onFocus={focus => schedule({ ...headControls, focus })}
              completed={display?.attentionRow && display.comparison.interventions[0]?.kind === 'attention.override' && activeHead
                ? { intervention: display.comparison.interventions[0], ...display.attentionRow } : undefined} />}
            <details className="attention-position-controls"><summary>Choose positions by number</summary>
            <label>Query position<select aria-label="Query position" disabled={locked} value={headControls.query} onChange={event => redirectQuery(Number(event.target.value))}>
              {tokenizer && tokenIds.map((id, i) => <option key={i} value={i}>Position {i}: {symbols![id]}</option>)}</select></label>
            <label>Attend to<select aria-label="Attend to" disabled={locked} value={headControls.focus} onChange={event => schedule({ ...headControls, focus: event.target.value })}>
              <option value="uniform">Evenly over positions 0–{headControls.query}</option>
              {tokenizer && tokenIds.slice(0, headControls.query + 1).map((id, i) => <option key={i} value={i}>All weight on position {i}: {symbols![id]}</option>)}</select></label>
            </details>
            <p className="hint">Replaces this head's attention probabilities for one query position after the softmax, then recomputes the head's output and everything downstream. The scores are unchanged; only later positions are excluded.</p>
          </div> : <>
          <label>Apply at<select aria-label="Apply at" disabled={loading || probing || transferring || !model} value={headControls.scope} onChange={event => schedule({ ...headControls, scope: event.target.value })}>
            <option value="all">All positions</option>{tokenizer && tokenIds.map((id, i) => <option key={i} value={i}>Position {i}: {symbols![id]}</option>)}</select></label>
          <div className="strength-presets" role="group" aria-label="Head strength presets">
            {[[0, 'Mute'], [1, 'Normal'], [2, 'Amplify']].map(([factor, label]) => <button type="button" key={label} aria-pressed={headControls.factor === factor}
              disabled={locked} onClick={() => schedule({ ...headControls, factor: Number(factor) })}>{label}<small>{factor}×</small></button>)}
          </div>
          <div className="strength-label"><label htmlFor="strength">Head strength</label><output htmlFor="strength">{headControls.factor.toFixed(2)}×</output></div>
          <input id="strength" type="range" min="0" max="2" step="any" value={headControls.factor} disabled={loading || probing || transferring || !model}
            onChange={event => schedule({ ...headControls, factor: Number(event.target.value) })} />
          <StrengthNumber key={model?.runtimeGeneration} value={headControls.factor} disabled={loading || probing || transferring || !model} onCommit={factor => schedule({ ...headControls, factor })} />
          <div className="scale-labels"><span>0 · ablated</span><span>1 · baseline</span><span>2 · doubled</span></div>
          </>}
          </>}
        </section>

          </> : <div className="node-navigation"><p className="hint">The detailed vectors and residual readouts are available below. Choose a head in the graph to change its contribution.</p>
            <button type="button" className="secondary" disabled={locked} onClick={() => setSelectedNode({ kind: 'head', layer: controls.layer, head: controls.head })}>Edit the active head</button></div>}
        </aside>
      </div>
      <div className="observation prediction-panel">
        <div className="step-heading"><span className="step-number">3</span><div><p className="eyebrow">Observe the consequence</p><h2>Compare the prediction</h2></div></div>
        {display && tokenizer ? <>
          <PredictionSummary comparison={display.comparison} target={display.controls.target} symbol={symbols![display.controls.target]}
            position={display.controls.position} pending={pending || loading || copying} ended={sessionEnded} disabled={locked}
            onReset={() => schedule({ ...controlsRef.current, mode: 'head', factor: 1 })} />
          <div className="prediction-title"><p>Next token after position <strong>{display.controls.position}</strong> · <SymbolLabel tokenizer={tokenizer} tokenId={display.tokens[display.controls.position]} /></p>
            <div className="legend"><span className="baseline-key">Baseline</span><span className="modified-key">Modified</span></div></div>
          <div className={pending ? 'probability-list stale' : 'probability-list'} aria-busy={pending}>
            {rows.map(row => <div className="probability-row" key={row.tokenId}><SymbolLabel tokenizer={tokenizer} tokenId={row.tokenId} />
              <div className="bars"><div className="bar baseline-bar" style={{ width: `${row.baselineProbability * 100}%` }} /><div className="bar modified-bar" style={{ width: `${row.probability * 100}%` }} /></div>
              <div className="numbers"><span>{(row.baselineProbability * 100).toFixed(2)}%</span><strong>{(row.probability * 100).toFixed(2)}%</strong></div>
              <span className={`delta ${row.probabilityDelta < 0 ? 'negative' : ''}`}>{row.probabilityDelta >= 0 ? '+' : ''}{(row.probabilityDelta * 100).toFixed(2)} pp</span></div>)}
          </div>
          <div className="metrics"><div><span>Distribution change</span><strong>{display.comparison.metrics.jsDivergence.toExponential(3)} <small>nats</small></strong></div>
            <div><label>Target symbol<select aria-label="Target symbol" value={controls.target} onChange={event => schedule({ ...controlsRef.current, target: Number(event.target.value) })} disabled={loading || probing || transferring}>
              {symbols!.map((symbol, i) => <option key={i} value={i}>{symbol}</option>)}</select></label><strong>{display.comparison.metrics.targetLogProbabilityDelta?.toFixed(4)} <small>Δ log p · {symbols![display.controls.target]}</small></strong></div>
            <div><span>Model computation</span><strong>{display.comparison.execution.elapsedMs.toFixed(1)} <small>ms</small></strong>
              <span data-testid="execution-strategy">{display.comparison.execution.strategy === 'cached-suffix'
                ? `Cached suffix · layer ${display.comparison.execution.recomputedFromLayer! + 1}`
                : display.comparison.execution.strategy === 'identity' ? 'Baseline reused' : 'Full forward pass'}</span></div></div>
          {!display.comparison.hasComputationalPath && <p className="path-note">This setting has no active computational path to the selected prediction. The baseline is preserved.</p>}

        </> : <div className="empty-observation">The computed prediction will appear here.</div>}
      </div>
    </div>
    <details className="workspace-utilities" open={view === 'details' ? true : undefined}><summary>Save, replay & runtime settings</summary><div className="workspace-tools">
        <section className="saved-experiments">
          <p className="eyebrow">Save and replay</p>
          <button className="secondary" disabled={loading || probing || transferring || pending || !display} onClick={() => void exportExperiment()}>Export experiment</button>
          <label className="file-label">Import experiment<input type="file" accept=".json,application/json" aria-label="Import experiment" disabled={loading || probing || transferring || pending || !model}
            onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void importExperiment(file); }} /></label>
          <p className="hint">Save completed settings and predictions as JSON. Import recomputes them using the exact installed checkpoint.{display?.controls.mode === 'patch' ? ' A saved patch includes its source input.' : ''}</p>
        </section>
        <section className="runtime">
          <p className="eyebrow">Computation runtime</p>
          <label>Backend<select aria-label="Backend" disabled={loading || probing || transferring} value={model?.backend ?? 'auto'} onChange={event => void load(event.target.value as 'cpu' | 'webgl', choice, text)}>
            <option value="auto">Initializing…</option><option value="cpu">CPU worker</option><option value="webgl">WebGL worker</option></select></label>
          <p className="hint">{model?.fallbackReason ?? (c ? `${c.layers} layers · ${c.heads} heads · ${c.modelDim} dimensions` : 'Loading static model assets…')}</p>
          {!model && !loading && <button className="secondary" onClick={() => void load('cpu', choice, text)}>Load with CPU</button>}
          <button className="secondary restart-worker" disabled={transferring} onClick={() => void restart()}>Restart computation worker on CPU</button>
          <button className="secondary" disabled={loading || probing || transferring || !model} onClick={() => void verify()}>{probing ? 'Checking computations…' : 'Verify against PyTorch'}</button>
          {probe && <div className="probe-result" role="status"><strong>Reference checks passed</strong><span>Forward error {probe.forwardMaximumError.toExponential(2)}</span>
            <span>Gradient error {probe.gradientMaximumError.toExponential(2)}</span><span>Adam error {probe.optimizerMaximumError.toExponential(2)}</span>
            <span>Retained tensor change: {probe.retainedTensorDelta}</span></div>}
        </section>

    </div>
    </details>
    {display && tokenizer && <section className="details-workspace" aria-label="Detailed computation inspection" hidden={view === 'visual'}>
      <div className="step-heading"><span className="step-number">＋</span><div><p className="eyebrow">Go deeper</p><h2>Inspect the numbers</h2></div></div>
          {experiments && <div className="checkpoint-evidence">
            <p><strong>{(experiments.evaluation.heldOut.accuracy * 100).toFixed(1)}% held-out answer accuracy</strong> · {experiments.evaluation.heldOut.count} distinct three-binding inputs</p>
            <p>Chance: 25% · local-context lookup: {(experiments.evaluation.localContext.accuracy * 100).toFixed(1)}%. Length tests: two bindings {(experiments.evaluation.lengthGeneralization['2'].accuracy * 100).toFixed(1)}%; four bindings {(experiments.evaluation.lengthGeneralization['4'].accuracy * 100).toFixed(1)}%.</p>
            <a href={`${import.meta.env.BASE_URL}models/retrieval/evaluation.json`} target="_blank" rel="noreferrer">Evaluation and limitations ↗</a>
          </div>}
          {display.replay && <div className={display.replay.matches ? 'replay-note' : 'error'} role="status" data-testid="replay-report">
            <strong>{display.replay.matches ? 'Replay matches saved results' : 'Replay differs from saved results'}</strong>
            <p>Saved {display.replay.savedBackend} → current {display.replay.currentBackend}. Maximum logit difference: baseline {display.replay.baselineMaximumError.toExponential(2)}, modified {display.replay.modifiedMaximumError.toExponential(2)}.</p>
            <span>Showing newly computed predictions. Comparison tolerance: 2e-5 absolute + 1e-5 relative.</span>
          </div>}
          {completedRecipe && <div className="recipe-note" data-testid="recipe-note"><strong>{completedRecipe.title}</strong><p>{completedRecipe.interpretation}</p>
            <span>Recorded PyTorch target probability: {(completedRecipe.metrics.baselineTargetProbability * 100).toFixed(2)}% → {(completedRecipe.metrics.modifiedTargetProbability * 100).toFixed(2)}%. Live browser results appear below.</span></div>}
          {display.controls.mode === 'patch' ? <div className="trace" aria-label="Intervention computation"><span>Source residual</span><span aria-hidden="true">→</span>
            <strong>{hookLabel(display.controls.hook)} · position {display.controls.patchPosition}</strong><span aria-hidden="true">→</span><span>Later layers</span><span aria-hidden="true">→</span><span>New prediction</span></div>
          : display.controls.mode === 'attention' ? <div className="trace" aria-label="Intervention computation"><strong>Layer {display.controls.layer + 1}, head {display.controls.head + 1} · query {display.controls.query} · {focusLabel(display.controls)}</strong>
            <span aria-hidden="true">→</span><span>Head output</span><span aria-hidden="true">→</span><span>Projection + residual</span><span aria-hidden="true">→</span><span>New prediction</span></div>
          : <div className="trace" aria-label="Intervention computation"><span>Attention weights</span><span aria-hidden="true">→</span>
            <strong>Head {display.controls.head + 1} · {display.controls.factor.toFixed(2)}×</strong><span aria-hidden="true">→</span><span>Projection + residual</span><span aria-hidden="true">→</span><span>New prediction</span></div>}

          {display.controls.mode === 'patch' ? <section className="patched-row"><p className="eyebrow">Patched residual</p><h3>The vector that was replaced</h3>
            <p className="hint">{hookLabel(display.controls.hook)}, position {display.controls.patchPosition}: this input's residual → the source input's residual. Everything after this point is recomputed.</p>
            <div className="vector-list">{display.raw.map((value, i) => <div key={i}><span>Dimension {i}</span><code>{value.toFixed(5)}</code><span aria-hidden="true">→</span><code>{display.modified[i].toFixed(5)}</code></div>)}</div>
          </section> : <div className="inspectors">
            <section><p className="eyebrow">Attention inspector</p><h3>{display.controls.mode === 'attention' ? 'Modified attention' : 'Where this head attends'}</h3><p className="hint">Rows are query positions; columns are source positions. {display.controls.mode === 'attention'
              ? `Row ${display.controls.query} is the replaced distribution; other rows are unchanged unless an earlier edit changed their inputs.` : 'Head-output scaling leaves these weights unchanged.'}</p>
              <div className="attention-scroll" tabIndex={0} aria-label="Scrollable attention matrix"><div className="attention-grid" style={{ gridTemplateColumns: `24px repeat(${length}, minmax(28px, 1fr))` }}><span />{display.tokens.map((id, i) => <span className="axis" key={`col-${i}`}>{i}</span>)}
                {display.tokens.map((_, query) => <div className="attention-row" key={query}><span className="axis">{query}</span>{display.tokens.map((id, key) => {
                  const value = display.attention[query * length + key];
                  return <button type="button" className={selectedCell?.query === query && selectedCell.source === key ? 'cell selected-cell' : 'cell'} key={key} disabled={loading || transferring} onClick={() => setSelectedCell({ query, source: key })} aria-pressed={selectedCell?.query === query && selectedCell.source === key} style={{ backgroundColor: `rgba(28, 116, 105, ${key > query ? 0 : .06 + value * .9})` }}
                    title={`Query ${query} → source ${key}: ${value.toFixed(6)}${key > query ? ' (causally masked)' : ''}`}
                    aria-label={`Query ${query}, source ${key}, weight ${value.toFixed(6)}${key > query ? ', masked' : ''}`}>{key > query ? '·' : value.toFixed(2)}</button>;
                })}</div>)}</div></div>
              {display.controls.mode === 'attention' ? <p className="hint">Cell arithmetic shows the softmax of the scores, which an override bypasses; switch to head scaling to inspect it.</p> : <>
              <p className="hint">Select a cell to inspect its Q/K dot product, causal mask, and softmax.</p>
              {selectedCell && client.current && <AttentionCellInspector client={client.current} runId={display.comparison.baselineRunId} layer={display.controls.layer} head={display.controls.head}
                query={selectedCell.query} source={selectedCell.source} weight={display.attention[selectedCell.query * length + selectedCell.source]} />}</>}
            </section>
            <section><p className="eyebrow">Head-output inspector</p><h3>The actual edited vector</h3><p className="hint">At position {display.controls.mode === 'attention' ? display.controls.query : display.controls.scope === 'all' ? display.controls.position : display.controls.scope}, before output projection. Raw output → modified output.</p>
              <div className="vector-list">{display.raw.map((value, i) => <div key={i}><span>Dimension {i}</span><code>{value.toFixed(5)}</code><span aria-hidden="true">→</span><code>{display.modified[i].toFixed(5)}</code></div>)}</div>
              <p className="hint">Changing this vector recomputes the residual, MLP, later layers, and output logits.</p>
            </section>
          </div>}
          {client.current && c && <ComputationInspector client={client.current} baselineRunId={display.comparison.baselineRunId} modifiedRunId={display.comparison.modifiedRunId}
            layer={display.controls.mode === 'patch' ? parseHook(display.controls.hook).layer : display.controls.layer}
            position={display.controls.mode === 'patch' ? display.controls.patchPosition : display.controls.mode === 'attention' ? display.controls.query : display.controls.scope === 'all' ? display.controls.position : Number(display.controls.scope)} layers={c.layers}
            symbols={display.tokens.map(id => symbols![id])} />}
          {client.current && <EmbeddingInspector client={client.current} runId={display.comparison.baselineRunId} position={display.controls.position}
            symbols={symbols!} inputSymbols={display.tokens.map(id => symbols![id])} />}
          {client.current && <ResidualLens client={client.current} baselineRunId={display.comparison.baselineRunId} modifiedRunId={display.comparison.modifiedRunId}
            position={display.controls.position} target={display.controls.target} symbols={symbols!} />}
          {client.current && <AblationExplorer client={client.current} baselineRunId={display.comparison.baselineRunId} position={display.controls.position} target={display.controls.target}
            symbols={symbols!} onApply={(layer, head) => schedule({ ...controlsRef.current, mode: 'head', layer, head, factor: 0, scope: 'all' })} />}

    </section>}
    <footer><span>Computed entirely in your browser.</span><span>Pre-LN · learned positions · tied embeddings · no dropout</span></footer>
  </main>;
}
