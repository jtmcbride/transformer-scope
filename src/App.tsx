import { useEffect, useRef, useState, type FormEvent } from 'react';
import { tokenize, type HeadScaleIntervention, type SymbolTokenizer } from './engine/contracts';
import type { ComparisonResult, RunSummary } from './engine/runs';
import type { Commands } from './workers/protocol';
import type { ProbeReport } from './engine/probe';
import { createExperiment, parseExperiment, validateExperiment, checkReplay, MAX_EXPERIMENT_BYTES, type ExperimentDocument, type ReplayReport } from './engine/experiment';
import { validateManifest, validateTokenizer } from './engine/weights';
import { ModelClient } from './workers/client';
import { fetchExperiments, validateExperiments, retrievalAnswer, type ModelChoice, type ExperimentMetadata, type ExperimentRecipe } from './experiments';

type LoadedModel = Commands['LOAD']['response'];
interface Controls { layer: number; head: number; factor: number; scope: string; position: number; target: number }
interface Display { comparison: ComparisonResult; controls: Controls; attention: number[]; raw: number[]; modified: number[]; tokens: number[]; replay?: ReplayReport }
interface Job { version: number; baseline: RunSummary; controls: Controls; replay?: ExperimentDocument }
const initialChoice: ModelChoice = new URLSearchParams(window.location.search).get('model') === 'verification' ? 'verification' : 'retrieval';
const defaults: Controls = { layer: 0, head: 0, factor: .5, scope: 'all', position: 5, target: 2 };

function SymbolLabel({ tokenizer, tokenId }: { tokenizer: SymbolTokenizer; tokenId: number }) {
  return <span className="symbol">{tokenizer.tokens[tokenId]}</span>;
}

export default function App() {
  const client = useRef<ModelClient | null>(null);
  const alive = useRef(true);
  const version = useRef(0);
  const baseline = useRef<RunSummary | null>(null);
  const latest = useRef<Job | null>(null);
  const running = useRef(false);
  const modifiedId = useRef<string | null>(null);
  const controlsRef = useRef<Controls>(defaults);
  const baselineInspector = useRef<{ key: string; attention: number[]; raw: number[] } | null>(null);
  const [choice, setChoice] = useState<ModelChoice>(initialChoice);
  const [experiments, setExperiments] = useState<ExperimentMetadata | null>(null);
  const [activeRecipe, setActiveRecipe] = useState<ExperimentRecipe | null>(null);
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

  async function drain() {
    if (running.current || !client.current) return;
    running.current = true;
    try {
      while (latest.current && alive.current) {
        const job: Job = latest.current;
        latest.current = null;
        const c = job.controls;
        const vectorPosition = c.scope === 'all' ? c.position : Number(c.scope);
        let created: string | undefined;
        try {
          const intervention: HeadScaleIntervention = { id: 'head-strength', kind: 'head.scale', layer: c.layer, head: c.head, factor: c.factor,
            tokens: c.scope === 'all' ? { kind: 'all' } : { kind: 'indices', indices: [Number(c.scope)] } };
          const comparison = await client.current.request('COMPARE', {
            baselineRunId: job.baseline.runId, interventions: [intervention], predictionPosition: c.position, targetTokenId: c.target, topK: 8,
          });
          created = comparison.modifiedRunId;
          if (job.version !== version.current) continue;
          const inspectorKey = JSON.stringify([job.baseline.runId, c.layer, c.head, vectorPosition]);
          const savedInspector = baselineInspector.current;
          const readBaseline = async () => {
            if (savedInspector?.key === inspectorKey) return savedInspector;
            const [attention, raw] = await Promise.all([
              client.current!.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'attention.weights', layer: c.layer }, selection: { head: c.head } }),
              client.current!.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'attention.head.raw', layer: c.layer }, selection: { head: c.head, position: vectorPosition } }),
            ]);
            return { key: inspectorKey, attention: Array.from(attention.values), raw: Array.from(raw.values) };
          };
          const [inspector, modified] = await Promise.all([
            readBaseline(),
            client.current.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'attention.head.modified', layer: c.layer }, selection: { head: c.head, position: vectorPosition } }),
          ]);
          if (job.version !== version.current) continue;
          let replay: ReplayReport | undefined;
          if (job.replay) {
            const [baseLogits, modifiedLogits] = await Promise.all([
              client.current.request('INSPECT', { runId: job.baseline.runId, reference: { stage: 'logits' }, selection: { position: c.position } }),
              client.current.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'logits' }, selection: { position: c.position } }),
            ]);
            if (job.version !== version.current) continue;
            replay = checkReplay(job.replay, baseLogits.values, modifiedLogits.values, comparison.execution.backend);
          }
          baselineInspector.current = inspector;
          if (modifiedId.current) void client.current.request('RELEASE', { runId: modifiedId.current }).catch(() => {});
          modifiedId.current = created; created = undefined;
          setDisplay({ comparison, controls: c, tokens: job.baseline.tokenIds,
            attention: inspector.attention, raw: inspector.raw, modified: Array.from(modified.values), replay });
          setError(null);
        } catch (failure) {
          if (alive.current && job.version === version.current) setError(failure instanceof Error ? failure.message : String(failure));
        } finally {
          if (created) void client.current.request('RELEASE', { runId: created }).catch(() => {});
        }
      }
    } finally { running.current = false; if (alive.current) setPending(false); }
  }

  function schedule(next: Controls, run = baseline.current, recipe: ExperimentRecipe | null = null, replay?: ExperimentDocument) {
    setActiveRecipe(recipe);
    controlsRef.current = next; setControls(next);
    const current = ++version.current;
    if (!run) return;
    latest.current = { version: current, baseline: run, controls: { ...next }, replay };
    setPending(true); void drain();
  }

  async function establish(loaded: LoadedModel, input: string, generation: number, recipe: ExperimentRecipe | null = null, replay?: ExperimentDocument) {
    const tokens = tokenize(input, loaded.tokenizer, loaded.manifest.config.contextLength);
    const next = await client.current!.request('BASELINE', { tokenIds: tokens });
    if (!alive.current || generation !== version.current) {
      void client.current?.request('RELEASE', { runId: next.runId }).catch(() => {}); return;
    }
    const old = baseline.current;
    baseline.current = next; setTokenIds(tokens);
    if (old) void client.current!.request('RELEASE', { runId: old.runId }).catch(() => {});
    if (modifiedId.current) {
      void client.current!.request('RELEASE', { runId: modifiedId.current }).catch(() => {}); modifiedId.current = null;
    }
    const patch = replay?.observation.interventions[0] ?? recipe?.intervention;
    schedule(patch ? { layer: patch.layer, head: patch.head, factor: patch.factor,
      scope: patch.tokens.kind === 'all' ? 'all' : String(patch.tokens.indices[0]), position: replay?.observation.predictionPosition ?? recipe!.predictionPosition, target: replay?.observation.targetTokenId ?? recipe!.targetTokenId }
      : { ...controlsRef.current, position: tokens.length - 1, scope: 'all', target: retrievalAnswer(tokens, loaded.tokenizer) ?? controlsRef.current.target }, next, recipe, replay);
  }

  async function load(backend: 'auto' | 'cpu' | 'webgl', selected: ModelChoice, input?: string, restored?: ExperimentDocument) {
    const generation = ++version.current;
    latest.current = null; baseline.current = null; modifiedId.current = null; baselineInspector.current = null;
    setChoice(selected); setExperiments(null); setActiveRecipe(null);
    setLoading(true); setModel(null); setDisplay(null); setError(null); setProbe(null);
    try {
      const manifestUrl = new URL(`${import.meta.env.BASE_URL}models/${selected}/manifest.json`, window.location.origin).href;
      const [loaded, data] = await Promise.all([
        client.current!.request('LOAD', { manifestUrl, backend }),
        selected === 'retrieval' ? fetchExperiments(manifestUrl) : Promise.resolve(null),
      ]);
      if (!alive.current || generation !== version.current) return;
      if (restored) validateExperiment(restored, loaded.manifest, loaded.tokenizer);
      const metadata = data ? validateExperiments(data, loaded.manifest, loaded.tokenizer) : null;
      const recipe = input === undefined ? metadata?.recipes[0] ?? null : null;
      const sequence = input ?? recipe?.text ?? 'A E I C G';
      if (input === undefined) controlsRef.current = defaults;
      setExperiments(metadata); setModel(loaded); setText(sequence);
      await establish(loaded, sequence, generation, recipe, restored);
    } catch (failure) {
      if (alive.current && generation === version.current) setError(failure instanceof Error ? failure.message : String(failure));
    } finally { if (alive.current) setLoading(false); }
  }

  async function selectRecipe(id: string) {
    const recipe = experiments?.recipes.find(item => item.id === id);
    if (!recipe || !model) return;
    const generation = ++version.current;
    latest.current = null; setLoading(true); setError(null); setText(recipe.text);
    try { await establish(model, recipe.text, generation, recipe); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setLoading(false); }
  }

  useEffect(() => {
    alive.current = true;
    client.current = new ModelClient();
    void load('auto', initialChoice);
    return () => { alive.current = false; ++version.current; latest.current = null; client.current?.dispose(); };
  }, []);

  async function runInput(event: FormEvent) {
    event.preventDefault();
    if (!model) return;
    const generation = ++version.current;
    latest.current = null; setLoading(true); setError(null);
    try { await establish(model, text, generation); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setLoading(false); }
  }

  async function verify() {
    setProbing(true); setError(null); setProbe(null);
    try { setProbe(await client.current!.request('PROBE', {})); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setProbing(false); }
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
        : document.checkpoint.modelId === 'verification-seed-1729' ? 'verification' : undefined;
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
  const rows = display ? display.comparison.displayTokens : [];
  const length = display?.tokens.length ?? 0;

  return <main>
    <header className="masthead">
      <div className="wordmark"><span className="mark" aria-hidden="true">μ</span><div><p className="eyebrow">An instrument for model experiments</p><h1>Transformer Microscope</h1></div></div>
      <span className="build-label">Controlled retrieval experiment</span>
    </header>
    <div className="notice"><strong>{choice === 'retrieval' ? 'Learned associative retrieval' : 'Verification model'}</strong>
      <span>{choice === 'retrieval' ? 'Read shuffled key–value bindings, then answer the query. Values change independently on every example. This model learns a small symbol task.' : 'Random weights, real computations. These predictions verify the engine; they do not demonstrate learned language.'}</span></div>
    <div className="workspace">
      <aside className="controls-panel">
        <section>
          <p className="eyebrow">Checkpoint</p>
          <label>Model<select aria-label="Model" disabled={loading || probing || transferring} value={choice} onChange={event => void load('auto', event.target.value as ModelChoice)}>
            <option value="retrieval">Trained retrieval model</option><option value="verification">Untrained verification model</option></select></label>
          {experiments && <>
            <label>Prepared experiment<select aria-label="Prepared experiment" disabled={loading || probing || transferring} value={activeRecipe?.id ?? ''} onChange={event => void selectRecipe(event.target.value)}>
              <option value="" disabled>Custom experiment</option>{experiments.recipes.map(recipe => <option key={recipe.id} value={recipe.id}>{recipe.title}</option>)}</select></label>
            <p className="hint">Each example was measured on this checkpoint. Adjust a control to explore further.</p>
          </>}
        </section>
        <section>
          <p className="eyebrow">Input sequence</p><h2>Set the observation</h2>
          <form onSubmit={runInput}><label htmlFor="sequence">Space-separated symbols</label>
            <textarea id="sequence" value={text} onChange={event => setText(event.target.value)} rows={2} spellCheck={false} disabled={transferring} />
            <p className="hint">{choice === 'verification' ? 'Use A–O. Up to 7 symbols, plus the beginning token.' : 'Use keys A–D, values 0–3, and separate = ; ? with spaces. Train format: three distinct bindings, then ? KEY =. Up to 31 symbols plus the beginning token.'}</p>
            <button type="submit" className="primary" disabled={loading || probing || transferring || !model}>Run input <span aria-hidden="true">→</span></button>
          </form>
          <div className="tokens" aria-label="Input tokens">{tokenizer && tokenIds.map((id, index) => <button key={index} disabled={loading || probing || transferring}
            className={controls.position === index ? 'token selected' : 'token'} onClick={() => schedule({ ...controlsRef.current, position: index })}
            aria-label={`Predict after position ${index}: ${tokenizer.tokens[id]}`} aria-pressed={controls.position === index}>
            <small>{index}</small>{tokenizer.tokens[id]}</button>)}</div>
        </section>
        <section>
          <p className="eyebrow">Intervention</p><h2>Change a head's contribution</h2>
          <div className="paired"><label>Layer<select aria-label="Layer" disabled={loading || probing || transferring || !c} value={controls.layer} onChange={event => schedule({ ...controlsRef.current, layer: Number(event.target.value) })}>
            {c && Array.from({ length: c.layers }, (_, i) => <option value={i} key={i}>Layer {i + 1}</option>)}</select></label>
            <label>Head<select aria-label="Head" disabled={loading || probing || transferring || !c} value={controls.head} onChange={event => schedule({ ...controlsRef.current, head: Number(event.target.value) })}>
              {c && Array.from({ length: c.heads }, (_, i) => <option value={i} key={i}>Head {i + 1}</option>)}</select></label></div>
          <label>Apply at<select aria-label="Apply at" disabled={loading || probing || transferring || !model} value={controls.scope} onChange={event => schedule({ ...controlsRef.current, scope: event.target.value })}>
            <option value="all">All positions</option>{tokenizer && tokenIds.map((id, i) => <option key={i} value={i}>Position {i}: {tokenizer.tokens[id]}</option>)}</select></label>
          <div className="strength-label"><label htmlFor="strength">Head strength</label><output htmlFor="strength">{controls.factor.toFixed(2)}×</output></div>
          <input id="strength" type="range" min="0" max="2" step="0.05" value={controls.factor} disabled={loading || probing || transferring || !model}
            onChange={event => schedule({ ...controlsRef.current, factor: Number(event.target.value) })} />
          <div className="scale-labels"><span>0 · ablated</span><span>1 · baseline</span><span>2 · doubled</span></div>
          <button className="secondary" disabled={loading || probing || transferring || !model} onClick={() => schedule({ ...controlsRef.current, factor: 1 })}>Reset to baseline</button>
        </section>
        <section className="saved-experiments">
          <p className="eyebrow">Save and replay</p>
          <button className="secondary" disabled={loading || probing || transferring || pending || !display} onClick={() => void exportExperiment()}>Export experiment</button>
          <label className="file-label">Import experiment<input type="file" accept=".json,application/json" aria-label="Import experiment" disabled={loading || probing || transferring || pending || !model}
            onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void importExperiment(file); }} /></label>
          <p className="hint">Save completed settings and predictions as JSON. Import recomputes them using the exact installed checkpoint.</p>
        </section>
        <section className="runtime">
          <p className="eyebrow">Computation runtime</p>
          <label>Backend<select aria-label="Backend" disabled={loading || probing || transferring} value={model?.backend ?? 'auto'} onChange={event => void load(event.target.value as 'cpu' | 'webgl', choice, text)}>
            <option value="auto">Initializing…</option><option value="cpu">CPU worker</option><option value="webgl">WebGL worker</option></select></label>
          <p className="hint">{model?.fallbackReason ?? (c ? `${c.layers} layers · ${c.heads} heads · ${c.modelDim} dimensions` : 'Loading static model assets…')}</p>
          {!model && !loading && <button className="secondary" onClick={() => void load('cpu', choice, text)}>Load with CPU</button>}
          <button className="secondary" disabled={loading || probing || transferring || !model} onClick={() => void verify()}>{probing ? 'Checking computations…' : 'Verify against PyTorch'}</button>
          {probe && <div className="probe-result" role="status"><strong>Reference checks passed</strong><span>Forward error {probe.forwardMaximumError.toExponential(2)}</span>
            <span>Gradient error {probe.gradientMaximumError.toExponential(2)}</span><span>Adam error {probe.optimizerMaximumError.toExponential(2)}</span>
            <span>Retained tensor change: {probe.retainedTensorDelta}</span></div>}
        </section>
      </aside>
      <div className="observation">
        <div className="observation-header"><div><p className="eyebrow">Prediction comparison</p><h2>Observe the consequence</h2></div>
          <span className={`status ${pending || loading ? 'working' : ''}`} role="status">{loading ? 'Loading model…' : pending ? `Computing…${display ? ` previous ${display.controls.factor.toFixed(2)}×` : ''}` : display ? `Showing ${display.controls.factor.toFixed(2)}×` : 'Ready'}</span></div>
        {error && <p className="error" role="alert">{error}</p>}
        {display && tokenizer ? <>
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
          {activeRecipe && <div className="recipe-note" data-testid="recipe-note"><strong>{activeRecipe.title}</strong><p>{activeRecipe.interpretation}</p>
            <span>Recorded PyTorch target probability: {(activeRecipe.metrics.baselineTargetProbability * 100).toFixed(2)}% → {(activeRecipe.metrics.modifiedTargetProbability * 100).toFixed(2)}%. Live browser results appear below.</span></div>}
          <div className="trace" aria-label="Intervention computation"><span>Attention weights</span><span aria-hidden="true">→</span>
            <strong>Head {display.controls.head + 1} · {display.controls.factor.toFixed(2)}×</strong><span aria-hidden="true">→</span><span>Projection + residual</span><span aria-hidden="true">→</span><span>New prediction</span></div>
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
              {tokenizer.tokens.map((symbol, i) => <option key={i} value={i}>{symbol}</option>)}</select></label><strong>{display.comparison.metrics.targetLogProbabilityDelta?.toFixed(4)} <small>Δ log p · {tokenizer.tokens[display.controls.target]}</small></strong></div>
            <div><span>Model computation</span><strong>{display.comparison.execution.elapsedMs.toFixed(1)} <small>ms</small></strong>
              <span data-testid="execution-strategy">{display.comparison.execution.strategy === 'cached-suffix'
                ? `Cached suffix · layer ${display.comparison.execution.recomputedFromLayer! + 1}`
                : display.comparison.execution.strategy === 'identity' ? 'Baseline reused' : 'Full forward pass'}</span></div></div>
          {!display.comparison.hasComputationalPath && <p className="path-note">This setting has no active computational path to the selected prediction. The baseline is preserved.</p>}
          <div className="inspectors">
            <section><p className="eyebrow">Attention inspector</p><h3>Where this head attends</h3><p className="hint">Rows are query positions; columns are source positions. Head-output scaling leaves these weights unchanged.</p>
              <div className="attention-scroll" tabIndex={0} aria-label="Scrollable attention matrix"><div className="attention-grid" style={{ gridTemplateColumns: `24px repeat(${length}, minmax(28px, 1fr))` }}><span />{display.tokens.map((id, i) => <span className="axis" key={`col-${i}`}>{i}</span>)}
                {display.tokens.map((_, query) => <div className="attention-row" key={query}><span className="axis">{query}</span>{display.tokens.map((id, key) => {
                  const value = display.attention[query * length + key];
                  return <button type="button" className="cell" key={key} style={{ backgroundColor: `rgba(28, 116, 105, ${key > query ? 0 : .06 + value * .9})` }}
                    title={`Query ${query} → source ${key}: ${value.toFixed(6)}${key > query ? ' (causally masked)' : ''}`}
                    aria-label={`Query ${query}, source ${key}, weight ${value.toFixed(6)}${key > query ? ', masked' : ''}`}>{key > query ? '·' : value.toFixed(2)}</button>;
                })}</div>)}</div></div>
            </section>
            <section><p className="eyebrow">Head-output inspector</p><h3>The actual edited vector</h3><p className="hint">At position {display.controls.scope === 'all' ? display.controls.position : display.controls.scope}, before output projection. Raw output → modified output.</p>
              <div className="vector-list">{display.raw.map((value, i) => <div key={i}><span>Dimension {i}</span><code>{value.toFixed(5)}</code><span aria-hidden="true">→</span><code>{display.modified[i].toFixed(5)}</code></div>)}</div>
              <p className="hint">Changing this vector recomputes the residual, MLP, later layers, and output logits.</p>
            </section>
          </div>
        </> : <div className="empty-observation">The model's computed predictions will appear here.</div>}
      </div>
    </div>
    <footer><span>Computed entirely in your browser.</span><span>Pre-LN · learned positions · tied embeddings · no dropout</span></footer>
  </main>;
}
