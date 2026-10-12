import { useEffect, useState } from 'react';
import type { ModelClient } from '../workers/client';
import type { ActivationStage } from '../engine/contracts';

const choices: { stage: ActivationStage; label: string; explanation: string }[] = [
  { stage: 'attention.projected', label: 'Attention output projection', explanation: 'All heads are concatenated and projected together, including the shared output bias.' },
  { stage: 'attention.residual', label: 'Residual after attention', explanation: 'The projected attention output is added to the block input.' },
  { stage: 'mlp.output', label: 'MLP output', explanation: 'Normalization, the first linear layer, GELU, and the second linear layer produce the MLP contribution.' },
  { stage: 'block.output', label: 'Block output', explanation: 'The MLP contribution is added to the attention residual. The next block receives this vector.' },
];
interface Props { client: ModelClient; baselineRunId: string; modifiedRunId: string; layer: number; position: number; layers: number; symbols: string[] }
export function ComputationInspector(props: Props) {
  const [open, setOpen] = useState(false);
  const [stage, setStage] = useState<ActivationStage>('attention.projected');
  const [layer, setLayer] = useState(props.layer);
  const [position, setPosition] = useState(props.position);
  const [slice, setSlice] = useState<{ key: string; baseline?: number[]; modified?: number[]; error?: string } | null>(null);
  useEffect(() => { setLayer(props.layer); setPosition(props.position); }, [props.layer, props.position, props.baselineRunId]);
  const valid = layer < props.layers && position < props.symbols.length;
  const key = JSON.stringify([props.baselineRunId, props.modifiedRunId, stage, layer, position]);
  useEffect(() => {
    if (!open || !valid) return;
    let current = true;
    const read = (runId: string) => props.client.request('INSPECT', { runId, reference: { stage, layer }, selection: { position } });
    void Promise.all([read(props.baselineRunId), read(props.modifiedRunId)]).then(([baseline, modified]) => {
      if (current) setSlice({ key, baseline: Array.from(baseline.values), modified: Array.from(modified.values) });
    }).catch(error => { if (current) setSlice({ key, error: error instanceof Error ? error.message : String(error) }); });
    return () => { current = false; };
  }, [props.client, props.baselineRunId, props.modifiedRunId, stage, layer, position, open, valid, key]);
  const current = slice?.key === key ? slice : null;
  return <section className="computation-inspector">
    <button type="button" className="secondary" aria-expanded={open} aria-controls="computation-details" onClick={() => setOpen(!open)}>{open ? 'Hide computation vectors' : 'Inspect projection, residual, and MLP'}</button>
    {open && <div id="computation-details">
      <p className="eyebrow">Computed stage vectors</p><h3>Follow the contribution through the block</h3>
      <div className="inspection-controls">
        <label>Computation stage<select value={stage} onChange={event => setStage(event.target.value as ActivationStage)}>{choices.map(choice => <option key={choice.stage} value={choice.stage}>{choice.label}</option>)}</select></label>
        <label>Inspect layer<select value={layer} onChange={event => setLayer(Number(event.target.value))}>{Array.from({ length: props.layers }, (_, i) => <option key={i} value={i}>Layer {i + 1}</option>)}</select></label>
        <label>Inspect position<select value={position} onChange={event => setPosition(Number(event.target.value))}>{props.symbols.map((symbol, i) => <option key={i} value={i}>{i}: {symbol}</option>)}</select></label>
      </div>
      <p className="hint">{choices.find(choice => choice.stage === stage)?.explanation} Values below belong to the completed experiment, at layer {layer + 1}, position {position}.</p>
      {current?.error ? <p className="error" role="alert">{current.error}</p> : !current?.baseline || !current.modified ? <p role="status">Reading computed vectors…</p> : <div className="vector-table-scroll"><table className="vector-table" aria-label="Baseline and modified computation vectors">
        <thead><tr><th>Dimension</th><th>Baseline</th><th>Modified</th><th>Change</th></tr></thead>
        <tbody>{current.baseline.map((value, i) => <tr key={i}><th scope="row">{i}</th><td>{value.toFixed(6)}</td><td>{current.modified![i].toFixed(6)}</td><td>{(current.modified![i] - value).toExponential(3)}</td></tr>)}</tbody>
      </table></div>}
    </div>}
  </section>;
}
