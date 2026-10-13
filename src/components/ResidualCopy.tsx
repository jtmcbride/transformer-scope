import type { ResidualReplaceIntervention } from '../engine/contracts';
import { hookKeys, hookLabel } from '../patching';

export interface CopyDraft { hook: string; position: number }
interface Props {
  draft: CopyDraft; sourceText: string; tokens: number[]; sourceTokens?: number[]; symbols: string[]; layers: number;
  disabled: boolean; pending: boolean; validation?: string; alternate?: string;
  completed?: { patch: ResidualReplaceIntervention; sourceTokens: number[]; baseline: number[]; copied: number[] };
  onSource: (text: string) => void; onDraft: (draft: CopyDraft) => void; onCopy: () => void; onClose: () => void;
}
const norm = (values: number[]) => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
export function ResidualCopy(props: Props) {
  const current = props.tokens.slice(1).map(token => props.symbols[token]).join(' ');
  return <section className="residual-copy" aria-label="Copy a residual vector">
    <div className="copy-heading"><div><p className="eyebrow">Copy a residual vector</p><h3>Source → current input</h3></div>
      <button type="button" className="text-button" disabled={props.disabled} onClick={props.onClose}>Close source editor</button></div>
    <p className="hint">Choose a source input, a junction in the flowchart, and a position. Copy from source applies one vector to the current input.</p>
    <label htmlFor="source">Source input (same number of symbols)</label>
    <textarea id="source" rows={2} spellCheck={false} value={props.sourceText} disabled={props.disabled} onChange={event => props.onSource(event.target.value)} />
    <div className="copy-source-presets">
      <button type="button" className="secondary" disabled={props.disabled} onClick={() => props.onSource(current)}>Use current input as source</button>
      {props.alternate && <button type="button" className="secondary" disabled={props.disabled} onClick={() => props.onSource(props.alternate!)}>Use alternate query</button>}
    </div>
    {props.validation ? <p className="copy-validation hint" role="status">{props.validation}</p> : <>
      <p className="eyebrow copy-position-label">Choose a position · Source above, current input below</p>
      <div className="copy-alignment" role="group" aria-label="Copy position">
        {props.tokens.map((token, position) => <button type="button" className={`copy-token-pair ${props.draft.position === position ? 'selected' : ''} ${props.sourceTokens?.[position] !== token ? 'different-symbol' : ''}`}
          key={position} disabled={props.disabled} aria-pressed={props.draft.position === position}
          aria-label={`Copy vector at position ${position}: ${props.symbols[props.sourceTokens![position]]} → ${props.symbols[token]}`}
          onClick={() => props.onDraft({ ...props.draft, position })}>
          <small>{position}</small><span className="copy-source-token">{props.symbols[props.sourceTokens![position]]}</span><span aria-hidden="true">↓</span><span>{props.symbols[token]}</span>
        </button>)}
      </div>
      <p className="hint">Marked pairs have different input symbols. Matching symbols can still have different residual vectors.</p>
    </>}
    <div className="copy-actions">
      <label>Copy junction<select aria-label="Patch point" disabled={props.disabled} value={props.draft.hook} onChange={event => props.onDraft({ ...props.draft, hook: event.target.value })}>
        {hookKeys(props.layers).map(hook => <option key={hook} value={hook}>{hookLabel(hook)}</option>)}
      </select></label>
      <button type="button" disabled={props.disabled || !!props.validation} onClick={props.onCopy}>Copy from source</button>
    </div>
    <p className="copy-draft" data-testid="copy-draft">Draft: {hookLabel(props.draft.hook)} · position {props.draft.position}. Prediction position stays selected above the graph.</p>
    <details className="copy-position-controls"><summary>Choose copy position by number</summary>
      <label>Copy position<select aria-label="Patch position" disabled={props.disabled} value={props.draft.position} onChange={event => props.onDraft({ ...props.draft, position: Number(event.target.value) })}>
        {props.tokens.map((token, position) => <option key={position} value={position}>Position {position}: {props.symbols[token]}{props.sourceTokens && props.sourceTokens[position] !== token ? ` ← ${props.symbols[props.sourceTokens[position]]}` : ''}</option>)}
      </select></label>
    </details>
    <p className="hint">Layer input can feed this layer's attention. After attention feeds the MLP; changes can spread to other positions through a later attention layer.</p>
    {props.completed && <div className="completed-copy" data-testid="completed-copy">
      <p className="eyebrow">{props.pending ? 'Last completed copy' : 'Completed copy'}</p>
      <p>{hookLabel(`${props.completed.patch.layer}:${props.completed.patch.stage}`)} · position {props.completed.patch.position} · source {props.symbols[props.completed.sourceTokens[props.completed.patch.position]]} → current {props.symbols[props.tokens[props.completed.patch.position]]}</p>
      <p className="copy-vector-norms">Original vector norm: {norm(props.completed.baseline).toFixed(4)} · Copied vector norm: {norm(props.completed.copied).toFixed(4)}</p>
      <p className="hint">The completed vector came from the source input recorded with this comparison. Editing the source above leaves it intact until the next copy.</p>
    </div>}
  </section>;
}
