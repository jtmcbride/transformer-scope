import { useEffect, useState } from 'react';
import type { ActivationReference } from '../engine/contracts';
import type { ModelClient } from '../workers/client';
import { nodeKey, stageNames, type FlowNode } from './ComputationGraph';
import { changeLabel, measurementFor, readoutFor, readoutProbabilities, traceNumber, type TraceView } from '../trace-view';
import { AttentionRowComparison, redirectConnection } from './AttentionRedirect';

const explanations = {
  'block.input': 'The residual stream entering this layer. It also travels along the bypass connection around attention.',
  'attention.normalized': 'Layer normalization prepares the residual for query, key, and value projections.',
  'attention.projected': 'All head outputs are joined and projected into the residual dimension, including the shared output bias.',
  'attention.residual': 'The projected attention output is added to the layer input at the first + junction.',
  'mlp.output': 'Normalization, a linear layer, GELU, and a second linear layer produce the MLP contribution at each position.',
  'block.output': 'The MLP contribution is added at the second + junction. This residual enters the next layer, or the final readout.',
};
const norm = (values: ArrayLike<number>) => Math.sqrt(Array.from(values).reduce((sum, value) => sum + value * value, 0));
interface Props { node: FlowNode; client: ModelClient; baselineRunId: string; modifiedRunId: string; position: number; symbol: string; sourceTokens?: number[]; preset?: { baseline: number[]; modified: number[] };
  trace?: TraceView; target?: number; symbols?: string[]; allowVector?: boolean }
export function NodeInspector(props: Props) {
  const [read, setRead] = useState<{ key: string; baseline?: number[]; modified?: number[]; error?: string } | null>(null);
  const key = JSON.stringify([nodeKey(props.node), props.baselineRunId, props.modifiedRunId, props.position]);
  const [vectorKey, setVectorKey] = useState('');
  const fullVector = vectorKey === key, measurement = props.trace ? measurementFor(props.trace, props.node) : undefined;
  const readout = props.trace ? readoutFor(props.trace, props.node) : undefined;
  const patch = props.trace?.report.intervention;
  const residual = patch?.kind === 'residual.replace' && props.node.kind === 'stage' && props.node.layer === patch.layer && props.node.stage === patch.stage && props.position === patch.position
    ? props.trace!.report.residualRow : undefined;
  useEffect(() => {
    if (props.preset || residual || measurement && !fullVector) return;
    let current = true;
    const node = props.node;
    const reference: ActivationReference = node.kind === 'embedding' ? { stage: 'embedding.output' } : node.kind === 'readout' ? { stage: 'logits' }
      : node.kind === 'head' ? { stage: node.part === 'attention' ? 'attention.weights' : 'attention.head.modified', layer: node.layer } : { stage: node.stage, layer: node.layer };
    const selection: Record<string, number> = node.kind === 'head' && node.part === 'attention' ? { queryPosition: props.position, head: node.head }
      : { position: props.position, ...(node.kind === 'head' ? { head: node.head } : {}) };
    void Promise.all([props.baselineRunId, props.modifiedRunId].map(runId => props.client.request('INSPECT', { runId, reference, selection })))
      .then(([baseline, modified]) => { if (current) setRead({ key, baseline: Array.from(baseline.values), modified: Array.from(modified.values) }); })
      .catch(error => { if (current) setRead({ key, error: error instanceof Error ? error.message : String(error) }); });
    return () => { current = false; };
  }, [props.client, props.baselineRunId, props.modifiedRunId, props.position, props.node, props.preset, key, measurement, fullVector, residual]);
  const current = residual ? { baseline: residual.baselineValues, modified: residual.copiedValues } : props.preset ?? (read?.key === key ? read : null);
  const baseline = current?.baseline, modified = current?.modified;
  const metrics = measurement ?? (baseline && modified ? { baselineNorm: norm(baseline), modifiedNorm: norm(modified),
    maxDifference: Math.max(...baseline.map((value, i) => Math.abs(value - modified[i]))) } : null);
  const explanation = props.node.kind === 'embedding' ? 'The token embedding and the learned position embedding are added to start the residual stream. These weights are shared by both runs.'
    : props.node.kind === 'readout' ? 'Final layer normalization and the transposed token embeddings produce logits. Softmax converts them to the next-token probabilities shown in the result card.'
    : props.node.kind === 'head' ? props.node.part === 'attention'
      ? 'These weights choose how much each source value contributes to this query. Redirecting replaces the weights after softmax; the scores and source value vectors in this layer stay unchanged.'
      : props.trace?.report.intervention.kind === 'attention.override'
      ? 'Attention weights combine source values into an output at each query. A redirect replaces one row of weights before this computation; an earlier-layer edit can also change later heads\' inputs and attention.'
      : props.trace ? 'Attention and head output are measured separately. Scaling this head acts after its attention weights are computed. An edit in an earlier layer can change this head\'s inputs and attention.'
      : 'This head combines value vectors using causal attention weights. Strength scales its output before the shared output projection. Scaling this head leaves its own attention weights unchanged; edits in earlier layers can change its inputs and attention.' : explanations[props.node.stage];
  return <div className="node-inspection">
    <p className="hint">{explanation}</p>
    <p className="slice-label">Completed vectors · position {props.position} · <span className="symbol">{props.symbol}</span>{props.node.kind === 'stage' ? ` · ${stageNames[props.node.stage]}` : ''}</p>
    {current && 'error' in current && current.error ? <p className="error" role="alert">{current.error}</p> : !metrics ? <p className="hint" role="status">Reading this node…</p> : <dl className="node-numbers" data-testid="node-values">
      <div><dt>Baseline norm</dt><dd>{metrics.baselineNorm.toFixed(4)}</dd></div>
      <div><dt>Modified norm</dt><dd>{metrics.modifiedNorm.toFixed(4)}</dd></div>
      <div><dt>Largest component change</dt><dd>{metrics.maxDifference.toExponential(2)}</dd></div>
      {measurement && <div><dt>L2 difference</dt><dd>{traceNumber(measurement.differenceNorm)}</dd></div>}
    </dl>}
    {measurement && <p className={`change-tag change-${measurement.change}`}>{changeLabel[measurement.change]}{measurement.change === 'small' ? ' · nonzero, within the display tolerance' : ''}</p>}
    {residual && <div className="copied-residual" data-testid="copied-residual">
      <p className="eyebrow">Completed vector copy · position {props.position}</p>
      {props.sourceTokens && props.symbols && <p>Source input: {props.symbols[props.sourceTokens[props.position]]} → current input: {props.symbol}</p>}
      <p>The source's residual vector at this junction and position replaced the current input's vector. The norms and differences above compare the original and copied vectors.</p>
      <p className="hint">{patch?.kind === 'residual.replace' && patch.stage === 'block.input' ? 'This copy enters attention in this layer and the residual bypass.' : 'This copy enters the MLP and its residual bypass. Attention in this layer was computed before the copy.'}</p>
    </div>}
    {props.node.kind === 'head' && props.node.part === 'attention' && props.trace?.report.intervention.kind === 'attention.override' && props.trace.report.attentionRow &&
      props.node.layer === props.trace.report.intervention.layer && props.node.head === props.trace.report.intervention.head && props.position === props.trace.report.intervention.query && props.symbols && <>
        <p className="redirect-connection">{redirectConnection(props.trace.report.intervention, props.trace.report.tokenIds, props.symbols)}</p>
        <p className="hint" data-testid="redirect-scores">Attention scores: {changeLabel[props.trace.report.attentionRow.scores.change]}. The edit replaces weights after softmax.</p>
        <AttentionRowComparison baseline={props.trace.report.attentionRow.baselineWeights} modified={props.trace.report.attentionRow.modifiedWeights}
          tokens={props.trace.report.tokenIds} symbols={props.symbols} query={props.position} />
      </>}
    {readout && props.symbols && <div className="trace-readout" data-testid="node-readout">
      <p className="eyebrow">{readout.prediction ? 'Model prediction' : 'Diagnostic readout'} · prediction position {props.trace!.report.predictionPosition}</p>
      <p>Preferred token: <strong>{props.symbols[readoutProbabilities(readout.baselineLogits, props.target ?? 0).top]} → {props.symbols[readoutProbabilities(readout.modifiedLogits, props.target ?? 0).top]}</strong></p>
      <p>p({props.symbols[props.target ?? 0]}): {(readoutProbabilities(readout.baselineLogits, props.target ?? 0).targetProbability * 100).toFixed(4)}% → {(readoutProbabilities(readout.modifiedLogits, props.target ?? 0).targetProbability * 100).toFixed(4)}%</p>
      <p data-testid="readout-delta">Δ log p: {traceNumber(readoutProbabilities(readout.modifiedLogits, props.target ?? 0).targetLogProbability - readoutProbabilities(readout.baselineLogits, props.target ?? 0).targetLogProbability)} nats</p>
      {!readout.prediction && <p className="hint">Final normalization and unembedding applied here for inspection. This is an intermediate residual readout.</p>}
    </div>}
    {(measurement || props.allowVector) && <button type="button" className="text-button" aria-expanded={fullVector} onClick={() => setVectorKey(fullVector ? '' : key)}>{fullVector ? 'Hide full vector' : 'Inspect full vector'}</button>}
    {(measurement || props.allowVector) && fullVector && baseline && modified && <div className="vector-table-scroll"><table className="vector-table" aria-label="Selected node vector dimensions">
      <thead><tr><th>Dim</th><th>Baseline</th><th>Modified</th><th>Difference</th></tr></thead>
      <tbody>{baseline.map((value, i) => <tr key={i}><th>{i}</th><td>{value.toFixed(6)}</td><td>{modified[i].toFixed(6)}</td><td>{traceNumber(modified[i] - value)}</td></tr>)}</tbody>
    </table></div>}
  </div>;
}
