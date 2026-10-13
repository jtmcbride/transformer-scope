import type { EffectTraceReport } from '../engine/trace';
import { firstReadoutFlip, handoff, readoutProbabilities, readoutTitle, type TraceStep } from '../trace-view';

interface Props { open: boolean; supported: boolean; busy: boolean; ended: boolean; report?: EffectTraceReport; error?: string; position: number; origin: number; steps: TraceStep[]; index: number; symbols: string[];
  onOpen: (open: boolean) => void; onRetry: () => void; onMove: (index: number) => void; onPosition: (position: number) => void }
export function EffectTrace(props: Props) {
  if (!props.open) return <button type="button" className="secondary follow-change" disabled={props.busy || !props.supported} onClick={() => props.onOpen(true)}>Follow the change <span aria-hidden="true">→</span></button>;
  const report = props.report, step = props.steps[props.index];
  const patch = report?.intervention;
  const flip = report ? firstReadoutFlip(report) : null, connection = report ? handoff(report, props.origin) : null;
  return <section className="effect-trace" aria-label="Effect trace">
    <div className="trace-heading"><p className="eyebrow">Follow the change</p></div>
    <div className="trace-navigation">
      <button type="button" className="secondary" disabled={!report || props.index <= 0} onClick={() => props.onMove(props.index - 1)}>Previous</button>
      <span aria-live="polite" data-testid="trace-step">{!report ? props.ended ? 'Worker session ended' : !props.supported ? 'Head scaling or attention redirect required' : props.error ? 'Trace unavailable' : 'Waiting for trace measurements…' : step ? `Step ${props.index + 1} of ${props.steps.length} · ${step.title} · position ${step.position}` : 'Manual inspection'}</span>
      <button type="button" className="secondary" disabled={!report || props.index >= props.steps.length - 1} onClick={() => props.onMove(props.index + 1)}>Next</button>
      <button type="button" className="text-button trace-exit" onClick={() => props.onOpen(false)}>Exit trace</button>
    </div>
    {!props.supported ? <p className="hint">Choose head scaling or an attention redirect to trace its downstream computation.</p>
      : props.error ? <div><p className="error" role="alert">{props.error}</p><button type="button" className="secondary" disabled={props.busy} onClick={props.onRetry}>Retry trace</button></div>
      : !report ? <p className="hint" role="status">{props.ended ? 'The worker session ended. Restart on CPU to measure a new trace.' : props.busy ? 'Waiting for the completed comparison…' : 'Measuring the completed trace…'}</p> : <>
        <p className="trace-scope">Layer {patch!.layer + 1} · Head {patch!.head + 1} · {patch!.kind === 'attention.override' ? `Redirect at query position ${patch!.query}`
          : `${patch!.factor}× · ${patch!.tokens.kind === 'all' ? patch!.factor === 1 ? 'All positions selected' : 'All positions edited' : `${patch!.factor === 1 ? 'Selected' : 'Edit'} position ${patch!.tokens.indices.join(', ')}`}`} · Prediction position {report.predictionPosition}</p>
        {patch!.kind === 'head.scale' && patch!.factor === 1 ? <p className="trace-path" data-testid="trace-path">Normal strength · no effective edit.</p>
          : !report.hasComputationalPath ? <p className="trace-path" data-testid="trace-path">Local changes have no computational path to prediction position {report.predictionPosition}.</p>
          : <p className="trace-path" data-testid="trace-path">Possible route to the prediction{connection ? ` through Layer ${connection.layer + 1} attention` : ''}. Measurements show what changed.{patch!.kind === 'head.scale' && patch!.tokens.kind === 'all' && props.origin > report.predictionPosition ? ` The guide starts after the prediction at position ${props.origin}; that local branch cannot reach it, although other edited positions can.` : ''}</p>}
        <label className="trace-position">Inspect position<select aria-label="Trace inspection position" value={props.position} onChange={event => props.onPosition(Number(event.target.value))}>
          {report.tokenIds.map((token, i) => <option value={i} key={i}>{i}: {props.symbols[token]}</option>)}</select></label>
        {flip && <p className="trace-flip" data-testid="trace-first-flip">The modified residual readout first prefers {props.symbols[readoutProbabilities(flip.modifiedLogits, 0).top]} instead of the baseline's {props.symbols[readoutProbabilities(flip.baselineLogits, 0).top]} here: {readoutTitle(flip)}{flip.prediction ? '' : ' · diagnostic readout'}. Position {report.predictionPosition}.</p>}
        <p className="trace-definition">Routes show possible data flow. Difference badges measure the inspected vector, including tiny nonzero changes.</p>
      </>}
  </section>;
}
