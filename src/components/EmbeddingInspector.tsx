import { useEffect, useState } from 'react';
import type { ModelClient } from '../workers/client';
import type { EmbeddingSlice } from '../engine/runs';

interface Props { client: ModelClient; runId: string; position: number; symbols: readonly string[]; inputSymbols: readonly string[] }
const norm = (values: Float32Array) => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));

export function EmbeddingInspector(props: Props) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState(props.position);
  const [read, setRead] = useState<{ key: string; slice?: EmbeddingSlice; error?: string } | null>(null);
  useEffect(() => setPosition(props.position), [props.position, props.runId]);
  const valid = position < props.inputSymbols.length;
  const key = JSON.stringify([props.runId, position]);
  useEffect(() => {
    if (!open || !valid) return;
    let current = true;
    props.client.request('EMBEDDING', { runId: props.runId, position })
      .then(slice => { if (current) setRead({ key, slice }); })
      .catch(error => { if (current) setRead({ key, error: error instanceof Error ? error.message : String(error) }); });
    return () => { current = false; };
  }, [props.client, props.runId, position, open, valid, key]);
  const slice = read?.key === key ? read.slice : undefined;
  const error = read?.key === key ? read.error : undefined;
  const ranked = slice ? Array.from(slice.similarity, (value, token) => ({ token, value })).filter(item => item.token !== slice.tokenId)
    .sort((a, b) => b.value - a.value || a.token - b.token) : [];
  const cosine = slice ? slice.token.reduce((sum, value, i) => sum + value * slice.positionEmbedding[i], 0) / (norm(slice.token) * norm(slice.positionEmbedding)) : 0;
  return <section className="residual-lens embedding-inspector">
    <button type="button" className="secondary" aria-expanded={open} aria-controls="embedding-details" onClick={() => setOpen(!open)}>{open ? 'Hide input embedding' : 'Inspect the input embedding'}</button>
    {open && <div id="embedding-details">
      <p className="eyebrow">Input embedding</p><h3>Where the residual stream starts</h3>
      <label>Embedding position<select value={position} onChange={event => setPosition(Number(event.target.value))}>
        {props.inputSymbols.map((symbol, i) => <option key={i} value={i}>{i}: {symbol}</option>)}</select></label>
      <p className="hint">Each position starts as its token's learned embedding plus a learned position embedding. These are weights, so they are identical in the baseline and modified runs; head scaling acts later. The same token embeddings are reused, transposed, to produce the output logits.</p>
      {error ? <p className="error" role="alert">{error}</p> : !slice ? <p role="status">Reading embeddings…</p> : <>
        <dl className="embedding-summary">
          <div><dt>Token</dt><dd><span className="symbol">{props.symbols[slice.tokenId]}</span> · norm {norm(slice.token).toFixed(4)}</dd></div>
          <div><dt>Position {slice.position}</dt><dd>norm {norm(slice.positionEmbedding).toFixed(4)}</dd></div>
          <div><dt>Sum</dt><dd>norm {norm(slice.sum).toFixed(4)}</dd></div>
          <div><dt>Token · position cosine</dt><dd>{cosine.toFixed(4)}</dd></div>
        </dl>
        <p className="hint">Cosine similarity of this token's embedding to every other token's embedding. Similar directions are a property of the learned weights, not a statement of meaning.</p>
        <div className="vector-table-scroll"><table className="vector-table" aria-label="Token embedding similarity">
          <thead><tr><th>Token</th><th>Cosine similarity</th></tr></thead>
          <tbody>{ranked.map(item => <tr key={item.token}><th scope="row">{props.symbols[item.token]}</th><td>{item.value.toFixed(4)}</td></tr>)}</tbody></table></div>
        <details><summary>Embedding vectors</summary><div className="vector-table-scroll"><table className="vector-table" aria-label="Token, position, and summed embedding vectors">
          <thead><tr><th>Dimension</th><th>Token</th><th>Position</th><th>Sum</th></tr></thead>
          <tbody>{Array.from(slice.sum, (value, i) => <tr key={i}><th scope="row">{i}</th><td>{slice.token[i].toFixed(6)}</td><td>{slice.positionEmbedding[i].toFixed(6)}</td><td>{value.toFixed(6)}</td></tr>)}</tbody></table></div></details>
      </>}
    </div>}
  </section>;
}
