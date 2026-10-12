import { useEffect, useState } from 'react';
import type { ModelClient } from '../workers/client';

interface Props { client: ModelClient; runId: string; layer: number; head: number; query: number; source: number; weight: number }
interface Cell { key: string; query: number[]; source: number[]; score: number; masked: number[]; error?: never }
export function AttentionCellInspector(props: Props) {
  const key = JSON.stringify([props.runId, props.layer, props.head, props.query, props.source]);
  const [cell, setCell] = useState<Cell | { key: string; error: string } | null>(null);
  useEffect(() => {
    let current = true;
    const base = { runId: props.runId };
    void Promise.all([
      props.client.request('INSPECT', { ...base, reference: { stage: 'attention.query', layer: props.layer }, selection: { head: props.head, position: props.query } }),
      props.client.request('INSPECT', { ...base, reference: { stage: 'attention.key', layer: props.layer }, selection: { head: props.head, position: props.source } }),
      props.client.request('INSPECT', { ...base, reference: { stage: 'attention.scores', layer: props.layer }, selection: { head: props.head, queryPosition: props.query, keyPosition: props.source } }),
      props.client.request('INSPECT', { ...base, reference: { stage: 'attention.maskedScores', layer: props.layer }, selection: { head: props.head, queryPosition: props.query } }),
    ]).then(([q, k, score, masked]) => { if (current) setCell({ key, query: Array.from(q.values), source: Array.from(k.values), score: score.values[0], masked: Array.from(masked.values) }); })
      .catch(error => { if (current) setCell({ key, error: error instanceof Error ? error.message : String(error) }); });
    return () => { current = false; };
  }, [props.client, props.runId, props.layer, props.head, props.query, props.source, key]);
  const current = cell?.key === key ? cell : null;
  if (!current) return <div className="cell-inspector" role="status">Reading attention-cell arithmetic…</div>;
  if (current.error !== undefined) return <p className="error" role="alert">{current.error}</p>;
  const dot = current.query.reduce((sum, value, i) => sum + value * current.source[i], 0);
  const maximum = Math.max(...current.masked);
  const denominator = current.masked.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  const numerator = Math.exp(current.masked[props.source] - maximum);
  const masked = props.source > props.query;
  return <div className="cell-inspector" data-testid="attention-cell-detail">
    <h3>One attention cell</h3><p className="hint">Layer {props.layer + 1}, head {props.head + 1}. Query {props.query} → source {props.source}. Head-output scaling does not change this same-layer calculation.</p>
    <dl><div><dt>Q · K from read-back vectors</dt><dd>{dot.toFixed(6)}</dd></div>
      <div><dt>Divide by √{current.query.length}</dt><dd>{(dot / Math.sqrt(current.query.length)).toFixed(6)}</dd></div>
      <div><dt>Computed score</dt><dd data-testid="cell-score">{current.score.toFixed(6)}</dd></div>
      <div><dt>Score after causal mask</dt><dd data-testid="cell-masked-score">{masked ? '-1e9 · future position' : current.masked[props.source].toFixed(6)}</dd></div>
      <div><dt>Stable softmax numerator / row sum</dt><dd>{numerator.toExponential(3)} / {denominator.toFixed(6)}</dd></div>
      <div><dt>Probability from masked row</dt><dd>{(numerator / denominator).toFixed(6)}</dd></div>
      <div><dt>Computed attention weight</dt><dd data-testid="cell-weight">{props.weight.toFixed(6)}</dd></div></dl>
    <p className="hint">The row calculation uses read-back float32 scores and double-precision arithmetic for explanation. Small rounding differences from the worker are expected.</p>
    <details><summary>Inspect Q/K products</summary><table className="vector-table" aria-label="Query and key dot products"><thead><tr><th>Dim</th><th>Q</th><th>K</th><th>Product</th></tr></thead>
      <tbody>{current.query.map((value, i) => <tr key={i}><th scope="row">{i}</th><td>{value.toFixed(5)}</td><td>{current.source[i].toFixed(5)}</td><td>{(value * current.source[i]).toFixed(5)}</td></tr>)}</tbody></table></details>
  </div>;
}
