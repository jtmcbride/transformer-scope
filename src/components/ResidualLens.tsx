import { useEffect, useState } from 'react';
import type { ModelClient } from '../workers/client';
import type { ResidualLens as Lens } from '../engine/runs';
import { logSoftmax } from '../engine/comparison';

interface Props { client: ModelClient; baselineRunId: string; modifiedRunId: string; position: number; target: number; symbols: readonly string[] }
interface Row { label: string; norm: number; top: number; topProbability: number; targetProbability: number }

function rows(lens: Lens, vocabulary: number, target: number): Row[] {
  return lens.stages.map((item, index) => {
    const probabilities = Array.from(logSoftmax(lens.logits.subarray(index * vocabulary, (index + 1) * vocabulary)), Math.exp);
    // Ties resolve to the lower token ID, as elsewhere in the screen.
    const top = probabilities.reduce((best, value, token) => value > probabilities[best] ? token : best, 0);
    const label = item.stage === 'embedding.output' ? 'Token + position embedding'
      : item.stage === 'attention.residual' ? `Layer ${item.layer! + 1} after attention` : `Layer ${item.layer! + 1} block output`;
    return { label, norm: lens.norms[index], top, topProbability: probabilities[top], targetProbability: probabilities[target] };
  });
}

export function ResidualLens(props: Props) {
  const [open, setOpen] = useState(false);
  const [read, setRead] = useState<{ key: string; baseline?: Lens; modified?: Lens; error?: string } | null>(null);
  const key = JSON.stringify([props.baselineRunId, props.modifiedRunId, props.position]);
  useEffect(() => {
    if (!open) return;
    let current = true;
    void Promise.all([props.baselineRunId, props.modifiedRunId].map(runId => props.client.request('LENS', { runId, position: props.position })))
      .then(([baseline, modified]) => { if (current) setRead({ key, baseline, modified }); })
      .catch(error => { if (current) setRead({ key, error: error instanceof Error ? error.message : String(error) }); });
    return () => { current = false; };
  }, [props.client, props.baselineRunId, props.modifiedRunId, props.position, open, key]);
  const current = read?.key === key ? read : null;
  const vocabulary = props.symbols.length;
  const baseline = current?.baseline ? rows(current.baseline, vocabulary, props.target) : null;
  const modified = current?.modified ? rows(current.modified, vocabulary, props.target) : null;
  const target = props.symbols[props.target];
  return <section className="residual-lens">
    <button type="button" className="secondary" aria-expanded={open} aria-controls="residual-lens" onClick={() => setOpen(!open)}>{open ? 'Hide residual stream' : 'Read out the residual stream'}</button>
    {open && <div id="residual-lens">
      <p className="eyebrow">Residual stream at position {props.position}</p><h3>What each stage would predict</h3>
      <p className="hint">Each residual vector is passed through the final normalization and the unembedding (a “logit lens”). Only the last row is the model's prediction. Earlier rows show what the residual would predict if the computation stopped there; the model does not compute them.</p>
      {current?.error ? <p className="error" role="alert">{current.error}</p> : !baseline || !modified ? <p role="status">Reading the residual stream…</p>
        : <div className="vector-table-scroll"><table className="vector-table lens-table" aria-label="Residual stream read-out">
          <thead><tr><th>Stage</th><th>Baseline top</th><th>Modified top</th><th>p({target}) baseline</th><th>p({target}) modified</th><th>Norm baseline</th><th>Norm modified</th></tr></thead>
          <tbody>{baseline.map((row, index) => <tr key={row.label} className={index === baseline.length - 1 ? 'final-row' : undefined}>
            <th scope="row">{row.label}{index === baseline.length - 1 ? ' · prediction' : ''}</th>
            <td>{props.symbols[row.top]} {(row.topProbability * 100).toFixed(2)}%</td><td>{props.symbols[modified[index].top]} {(modified[index].topProbability * 100).toFixed(2)}%</td>
            <td>{(row.targetProbability * 100).toFixed(2)}%</td><td>{(modified[index].targetProbability * 100).toFixed(2)}%</td>
            <td>{row.norm.toFixed(3)}</td><td>{modified[index].norm.toFixed(3)}</td></tr>)}</tbody></table></div>}
    </div>}
  </section>;
}
