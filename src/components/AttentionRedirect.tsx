import type { AttentionOverrideIntervention } from '../engine/contracts';

export function redirectConnection(patch: AttentionOverrideIntervention, tokens: number[], symbols: string[]) {
  const sources = patch.weights.flatMap((weight, position) => weight > 0 ? [position] : []);
  const source = sources.length === 1 ? `Source ${sources[0]}: ${symbols[tokens[sources[0]]]}`
    : sources.length === patch.query + 1 ? `Sources 0–${patch.query}` : `Sources ${sources.join(', ')}`;
  return `${source} → Query ${patch.query}: ${symbols[tokens[patch.query]]}`;
}

export function AttentionRowComparison({ baseline, modified, tokens, symbols, query }: {
  baseline: number[]; modified: number[]; tokens: number[]; symbols: string[]; query: number;
}) {
  return <div className="attention-row-comparison">
    <p className="eyebrow">Measured weights · query position {query}</p>
    <div className="attention-row-scroll"><table aria-label="Attention weights before and after redirect">
      <thead><tr><th>Source</th><th>Baseline</th><th>Modified</th></tr></thead>
      <tbody>{tokens.map((token, position) => <tr key={position} data-source-position={position}>
        <th>{position}: {symbols[token]}{position > query && <small>Future · excluded</small>}</th>
        <td>{(baseline[position] * 100).toFixed(2)}%</td><td>{(modified[position] * 100).toFixed(2)}%</td>
      </tr>)}</tbody>
    </table></div>
  </div>;
}

interface Props {
  tokens: number[]; symbols: string[]; query: number; focus: string; disabled: boolean; pending: boolean;
  showWeights?: boolean;
  completed?: { intervention: AttentionOverrideIntervention; baseline: number[]; modified: number[] };
  onQuery: (position: number) => void; onFocus: (focus: string) => void;
}
export function AttentionRedirect(props: Props) {
  return <div className="attention-redirect" aria-label="Redirect attention">
    <p className="hint">Choose the query token whose attention to redirect, then a source token for it to read.</p>
    <div className="attention-token-group" role="group" aria-label="Query token">
      <p className="eyebrow">1 · Query token</p>
      <div className="attention-tokens">{props.tokens.map((token, position) => <button type="button" key={position}
        className={`token ${props.query === position ? 'selected' : ''}`} disabled={props.disabled} aria-pressed={props.query === position}
        aria-label={`Query position ${position}: ${props.symbols[token]}`} onClick={() => props.onQuery(position)}>
        <small>{position}</small>{props.symbols[token]}
      </button>)}</div>
    </div>
    <div className="attention-token-group" role="group" aria-label="Attention source">
      <p className="eyebrow">2 · Attend to source</p>
      <button type="button" className="secondary uniform-attention" disabled={props.disabled} aria-pressed={props.focus === 'uniform'}
        onClick={() => props.onFocus('uniform')}>Spread evenly over positions 0–{props.query}</button>
      <div className="attention-tokens">{props.tokens.map((token, position) => <button type="button" key={position}
        className={`token ${props.focus === String(position) ? 'selected' : ''}`} disabled={props.disabled || position > props.query}
        aria-pressed={props.focus === String(position)} aria-label={`Attend to position ${position}: ${props.symbols[token]}${position > props.query ? ' (future token unavailable)' : ''}`}
        title={position > props.query ? 'A query can only read itself and earlier tokens.' : undefined} onClick={() => props.onFocus(String(position))}>
        <small>{position}</small>{props.symbols[token]}{position > props.query && <small>Future</small>}
      </button>)}</div>
    </div>
    <p className="hint">Future tokens are unavailable. Each edit changes one query's attention weights; the prediction position stays selected above the graph.</p>
    {props.completed && <div className="completed-redirect">
      <p className="eyebrow">{props.pending ? 'Last completed redirect' : 'Completed redirect'} · Layer {props.completed.intervention.layer + 1} · Head {props.completed.intervention.head + 1}</p>
      <p className="redirect-connection" data-testid="redirect-connection">{redirectConnection(props.completed.intervention, props.tokens, props.symbols)}</p>
      {props.showWeights !== false && <AttentionRowComparison baseline={props.completed.baseline} modified={props.completed.modified} tokens={props.tokens} symbols={props.symbols} query={props.completed.intervention.query} />}
    </div>}
  </div>;
}
