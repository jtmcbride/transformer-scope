import { useEffect, useRef, useState, type FormEvent } from 'react';
import { tokenize, type HeadScaleIntervention, type SymbolTokenizer } from './engine/contracts';
import type { ComparisonResult, RunSummary } from './engine/runs';
import type { Commands } from './workers/protocol';
import type { ProbeReport } from './engine/probe';
import { ModelClient } from './workers/client';

type LoadedModel = Commands['LOAD']['response'];
interface Controls { layer: number; head: number; factor: number; scope: string; position: number; target: number }
interface Display { comparison: ComparisonResult; controls: Controls; attention: number[]; raw: number[]; modified: number[]; tokens: number[] }
interface Job { version: number; baseline: RunSummary; controls: Controls }
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
          baselineInspector.current = inspector;
          if (modifiedId.current) void client.current.request('RELEASE', { runId: modifiedId.current }).catch(() => {});
          modifiedId.current = created; created = undefined;
          setDisplay({ comparison, controls: c, tokens: job.baseline.tokenIds,
            attention: inspector.attention, raw: inspector.raw, modified: Array.from(modified.values) });
          setError(null);
        } catch (failure) {
          if (alive.current && job.version === version.current) setError(failure instanceof Error ? failure.message : String(failure));
        } finally {
          if (created) void client.current.request('RELEASE', { runId: created }).catch(() => {});
        }
      }
    } finally { running.current = false; if (alive.current) setPending(false); }
  }

  function schedule(next: Controls, run = baseline.current) {
    controlsRef.current = next; setControls(next);
    const current = ++version.current;
    if (!run) return;
    latest.current = { version: current, baseline: run, controls: { ...next } };
    setPending(true); void drain();
  }

  async function establish(loaded: LoadedModel, input: string, generation: number) {
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
    schedule({ ...controlsRef.current, position: tokens.length - 1, scope: 'all' }, next);
  }

  async function load(backend: 'auto' | 'cpu' | 'webgl', input: string) {
    const generation = ++version.current;
    latest.current = null; baseline.current = null; modifiedId.current = null; baselineInspector.current = null;
    setLoading(true); setModel(null); setDisplay(null); setError(null); setProbe(null);
    try {
      const loaded = await client.current!.request('LOAD', {
        manifestUrl: new URL(`${import.meta.env.BASE_URL}models/verification/manifest.json`, window.location.origin).href, backend,
      });
      if (!alive.current || generation !== version.current) return;
      setModel(loaded); await establish(loaded, input, generation);
    } catch (failure) {
      if (alive.current && generation === version.current) setError(failure instanceof Error ? failure.message : String(failure));
    } finally { if (alive.current) setLoading(false); }
  }

  useEffect(() => {
    alive.current = true;
    client.current = new ModelClient();
    void load('auto', 'A E I C G');
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

  const tokenizer = model?.tokenizer;
  const c = model?.manifest.config;
  const rows = display ? display.comparison.displayTokens : [];
  const length = display?.tokens.length ?? 0;

  return <main>
    <header className="masthead">
      <div className="wordmark"><span className="mark" aria-hidden="true">μ</span><div><p className="eyebrow">An instrument for model experiments</p><h1>Transformer Microscope</h1></div></div>
      <span className="build-label">First computation milestone</span>
    </header>
    <div className="notice"><strong>Verification model</strong><span>Random weights, real computations. These predictions verify the engine; they do not demonstrate learned language.</span></div>
    <div className="workspace">
      <aside className="controls-panel">
        <section>
          <p className="eyebrow">Input sequence</p><h2>Set the observation</h2>
          <form onSubmit={runInput}><label htmlFor="sequence">Space-separated symbols</label>
            <textarea id="sequence" value={text} onChange={event => setText(event.target.value)} rows={2} spellCheck={false} />
            <p className="hint">Use A–O. Up to 7 symbols, plus the beginning token.</p>
            <button type="submit" className="primary" disabled={loading || !model}>Run input <span aria-hidden="true">→</span></button>
          </form>
          <div className="tokens" aria-label="Input tokens">{tokenizer && tokenIds.map((id, index) => <button key={index} disabled={loading}
            className={controls.position === index ? 'token selected' : 'token'} onClick={() => schedule({ ...controlsRef.current, position: index })}
            aria-label={`Predict after position ${index}: ${tokenizer.tokens[id]}`} aria-pressed={controls.position === index}>
            <small>{index}</small>{tokenizer.tokens[id]}</button>)}</div>
        </section>
        <section>
          <p className="eyebrow">Intervention</p><h2>Change a head's contribution</h2>
          <div className="paired"><label>Layer<select aria-label="Layer" disabled={loading || !c} value={controls.layer} onChange={event => schedule({ ...controlsRef.current, layer: Number(event.target.value) })}>
            {c && Array.from({ length: c.layers }, (_, i) => <option value={i} key={i}>Layer {i + 1}</option>)}</select></label>
            <label>Head<select aria-label="Head" disabled={loading || !c} value={controls.head} onChange={event => schedule({ ...controlsRef.current, head: Number(event.target.value) })}>
              {c && Array.from({ length: c.heads }, (_, i) => <option value={i} key={i}>Head {i + 1}</option>)}</select></label></div>
          <label>Apply at<select aria-label="Apply at" disabled={loading || !model} value={controls.scope} onChange={event => schedule({ ...controlsRef.current, scope: event.target.value })}>
            <option value="all">All positions</option>{tokenizer && tokenIds.map((id, i) => <option key={i} value={i}>Position {i}: {tokenizer.tokens[id]}</option>)}</select></label>
          <div className="strength-label"><label htmlFor="strength">Head strength</label><output htmlFor="strength">{controls.factor.toFixed(2)}×</output></div>
          <input id="strength" type="range" min="0" max="2" step="0.05" value={controls.factor} disabled={loading || !model}
            onChange={event => schedule({ ...controlsRef.current, factor: Number(event.target.value) })} />
          <div className="scale-labels"><span>0 · ablated</span><span>1 · baseline</span><span>2 · doubled</span></div>
          <button className="secondary" disabled={loading || !model} onClick={() => schedule({ ...controlsRef.current, factor: 1 })}>Reset to baseline</button>
        </section>
        <section className="runtime">
          <p className="eyebrow">Computation runtime</p>
          <label>Backend<select aria-label="Backend" disabled={loading || probing} value={model?.backend ?? 'auto'} onChange={event => void load(event.target.value as 'cpu' | 'webgl', text)}>
            <option value="auto">Initializing…</option><option value="cpu">CPU worker</option><option value="webgl">WebGL worker</option></select></label>
          <p className="hint">{model?.fallbackReason ?? (c ? `${c.layers} layers · ${c.heads} heads · ${c.modelDim} dimensions` : 'Loading static model assets…')}</p>
          {!model && !loading && <button className="secondary" onClick={() => void load('cpu', text)}>Load with CPU</button>}
          <button className="secondary" disabled={loading || probing || !model} onClick={() => void verify()}>{probing ? 'Checking computations…' : 'Verify against PyTorch'}</button>
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
            <div><label>Target symbol<select aria-label="Target symbol" value={controls.target} onChange={event => schedule({ ...controlsRef.current, target: Number(event.target.value) })} disabled={loading}>
              {tokenizer.tokens.map((symbol, i) => <option key={i} value={i}>{symbol}</option>)}</select></label><strong>{display.comparison.metrics.targetLogProbabilityDelta?.toFixed(4)} <small>Δ log p · {tokenizer.tokens[display.controls.target]}</small></strong></div>
            <div><span>Model computation</span><strong>{display.comparison.execution.elapsedMs.toFixed(1)} <small>ms</small></strong>
              <span data-testid="execution-strategy">{display.comparison.execution.strategy === 'cached-suffix'
                ? `Cached suffix · layer ${display.comparison.execution.recomputedFromLayer! + 1}`
                : display.comparison.execution.strategy === 'identity' ? 'Baseline reused' : 'Full forward pass'}</span></div></div>
          {!display.comparison.hasComputationalPath && <p className="path-note">This setting has no active computational path to the selected prediction. The baseline is preserved.</p>}
          <div className="inspectors">
            <section><p className="eyebrow">Attention inspector</p><h3>Where this head attends</h3><p className="hint">Rows are query positions; columns are source positions. Head-output scaling leaves these weights unchanged.</p>
              <div className="attention-grid" style={{ gridTemplateColumns: `24px repeat(${length}, 1fr)` }}><span />{display.tokens.map((id, i) => <span className="axis" key={`col-${i}`}>{i}</span>)}
                {display.tokens.map((_, query) => <div className="attention-row" key={query}><span className="axis">{query}</span>{display.tokens.map((id, key) => {
                  const value = display.attention[query * length + key];
                  return <button type="button" className="cell" key={key} style={{ backgroundColor: `rgba(28, 116, 105, ${key > query ? 0 : .06 + value * .9})` }}
                    title={`Query ${query} → source ${key}: ${value.toFixed(6)}${key > query ? ' (causally masked)' : ''}`}
                    aria-label={`Query ${query}, source ${key}, weight ${value.toFixed(6)}${key > query ? ', masked' : ''}`}>{key > query ? '·' : value.toFixed(2)}</button>;
                })}</div>)}</div>
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
