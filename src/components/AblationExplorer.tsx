import { useEffect, useState } from 'react';
import type { ModelClient } from '../workers/client';
import type { HeadAblationScan } from '../engine/runs';
import type { HeadScaleIntervention } from '../engine/contracts';

interface Props { client: ModelClient; baselineRunId: string; position: number; target: number; symbols: readonly string[]; onApply: (layer: number, head: number) => void }
const name = (layer: number, head: number) => `Layer ${layer + 1}, Head ${head + 1}`;
const ablate = (layer: number, head: number): HeadScaleIntervention => ({ id: `ablate-${layer}-${head}`, kind: 'head.scale', layer, head, factor: 0, tokens: { kind: 'all' } });

export function AblationExplorer(props: Props) {
  const [open, setOpen] = useState(false);
  const key = JSON.stringify([props.baselineRunId, props.position, props.target]);
  const [scan, setScan] = useState<{ key: string; result?: HeadAblationScan; error?: string } | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [pair, setPair] = useState<{ key: string; heads: string[]; joint?: number; error?: string } | null>(null);
  useEffect(() => { setSelected([]); setPair(null); }, [key]);
  useEffect(() => {
    if (!open) return;
    let current = true;
    props.client.request('ABLATION_SCAN', { baselineRunId: props.baselineRunId, predictionPosition: props.position, targetTokenId: props.target })
      .then(result => { if (current) setScan({ key, result }); })
      .catch(error => { if (current) setScan({ key, error: error instanceof Error ? error.message : String(error) }); });
    return () => { current = false; };
  }, [props.client, props.baselineRunId, props.position, props.target, open, key]);
  const result = scan?.key === key ? scan.result : undefined;
  const error = scan?.key === key ? scan.error : undefined;
  const rows = result ? [...result.entries].sort((a, b) => a.metrics.targetLogProbabilityDelta! - b.metrics.targetLogProbabilityDelta! || a.layer - b.layer || a.head - b.head) : [];
  const largest = Math.max(1e-12, ...rows.map(row => Math.abs(row.metrics.targetLogProbabilityDelta!)));
  const delta = (id: string) => result!.entries.find(entry => `${entry.layer}:${entry.head}` === id)!.metrics.targetLogProbabilityDelta!;
  const baselineTop = result ? Array.from(result.baselineLogits).reduce((best, value, token, all) => value > all[best] ? token : best, 0) : 0;
  const target = props.symbols[props.target];

  async function ablatePair() {
    const heads = [...selected], requestKey = key;
    setPair({ key: requestKey, heads });
    try {
      const comparison = await props.client.request('COMPARE', { baselineRunId: props.baselineRunId, predictionPosition: props.position, targetTokenId: props.target, topK: 1,
        interventions: heads.map(id => { const [layer, head] = id.split(':').map(Number); return ablate(layer, head); }) });
      void props.client.request('RELEASE', { runId: comparison.modifiedRunId }).catch(() => {});
      setPair(previous => previous?.key === requestKey ? { key: requestKey, heads, joint: comparison.metrics.targetLogProbabilityDelta } : previous);
    } catch (failure) {
      setPair(previous => previous?.key === requestKey ? { key: requestKey, heads, error: failure instanceof Error ? failure.message : String(failure) } : previous);
    }
  }

  const shownPair = pair?.key === key ? pair : null;
  return <section className="residual-lens ablation-explorer">
    <button type="button" className="secondary" aria-expanded={open} aria-controls="ablation-details" onClick={() => setOpen(!open)}>{open ? 'Hide head ablations' : 'Ablate each head for this prediction'}</button>
    {open && <div id="ablation-details">
      <p className="eyebrow">Ablation explorer · position {props.position} · target {target}</p><h3>Which heads this prediction depends on</h3>
      <p className="hint">Each row sets one head's output to 0 at every position, from the same baseline, and records the change in log p({target}). These are measurements for this input, prediction position, and target only; other inputs can rank heads differently, and ablation does not reveal what a head represents. Effects of several heads do not simply add: select two heads to compare.</p>
      {error ? <p className="error" role="alert">{error}</p> : !result ? <p role="status">Ablating each head…</p> : <>
        <div className="vector-table-scroll"><table className="vector-table ablation-table" aria-label="Single-head ablations">
          <thead><tr><th>Head</th><th>Δ log p({target})</th><th>p({target}) after</th><th>Distribution change (nats)</th><th>Top prediction after</th><th>Pair</th><th></th></tr></thead>
          <tbody>{rows.map(row => {
            const id = `${row.layer}:${row.head}`, change = row.metrics.targetLogProbabilityDelta!;
            return <tr key={id}><th scope="row">{name(row.layer, row.head)}</th>
              <td><span className="effect-bar" style={{ width: `${Math.abs(change) / largest * 60}px` }} aria-hidden="true" />{change.toFixed(4)}</td>
              <td>{(row.targetProbability * 100).toFixed(2)}%</td><td>{row.metrics.jsDivergence.toExponential(2)}</td>
              <td>{props.symbols[row.topTokenId]}{row.topTokenId !== baselineTop ? ' · changed' : ''}</td>
              <td><input type="checkbox" aria-label={`Select ${name(row.layer, row.head)} for a pair`} checked={selected.includes(id)}
                disabled={!selected.includes(id) && selected.length >= 2} onChange={event => { setPair(null); setSelected(event.target.checked ? [...selected, id] : selected.filter(item => item !== id)); }} /></td>
              <td><button type="button" className="secondary compact" onClick={() => props.onApply(row.layer, row.head)} aria-label={`Show ${name(row.layer, row.head)} ablated`}>Show</button></td></tr>;
          })}</tbody></table></div>
        <div className="pair-controls">
          <button type="button" className="secondary" disabled={selected.length !== 2 || (shownPair !== null && shownPair.joint === undefined && !shownPair.error)} onClick={() => void ablatePair()}>Ablate the selected pair together</button>
          {shownPair?.error ? <p className="error" role="alert">{shownPair.error}</p> : shownPair?.joint !== undefined && <p className="pair-result" data-testid="pair-result">
            {shownPair.heads.map(id => name(...id.split(':').map(Number) as [number, number])).join(' + ')}: together Δ log p = <strong>{shownPair.joint.toFixed(4)}</strong>;
            {' '}sum of single effects = <strong>{(delta(shownPair.heads[0]) + delta(shownPair.heads[1])).toFixed(4)}</strong>;
            {' '}difference = <strong>{(shownPair.joint - delta(shownPair.heads[0]) - delta(shownPair.heads[1])).toFixed(4)}</strong>.</p>}
        </div>
      </>}
    </div>}
  </section>;
}
