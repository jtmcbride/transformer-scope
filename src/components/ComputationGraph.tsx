import { useEffect, useState } from 'react';
import type { Intervention, ModelConfig } from '../engine/contracts';
import { nodeKey, stageNames, type FlowNode } from '../flow';
import { changeLabel, handoff, measurementFor, readoutFor, readoutProbabilities, routeLabel, traceNumber, type TraceView } from '../trace-view';
export { nodeKey, stageNames, type FlowNode, type FlowStage } from '../flow';

interface Props { config: ModelConfig; selected: FlowNode; intervention?: Intervention; disabled: boolean; onSelect: (node: FlowNode) => void; trace?: TraceView; symbols?: string[]; target?: number }
export function ComputationGraph(props: Props) {
  const [collapsed, setCollapsed] = useState<number[]>([]);
  const selectedKey = nodeKey(props.selected);
  useEffect(() => {
    if (props.trace && (props.selected.kind === 'head' || props.selected.kind === 'stage')) {
      const layer = props.selected.layer;
      setCollapsed(previous => previous.filter(item => item !== layer));
    }
  }, [selectedKey, !!props.trace, props.trace?.position]);
  const activeKey = props.intervention?.kind === 'residual.replace' ? `${props.intervention.layer}:${props.intervention.stage}`
    : props.intervention && (props.intervention.kind !== 'head.scale' || props.intervention.factor !== 1) ? `head:${props.intervention.layer}:${props.intervention.head}` : null;
  function node(selection: FlowNode, title: string, detail?: string, extra = '') {
    const key = nodeKey(selection), edited = key === activeKey;
    const measured = props.trace ? measurementFor(props.trace, selection) : undefined;
    const weights = props.trace && selection.kind === 'head' ? measurementFor(props.trace, selection, true) : undefined;
    const readout = props.trace ? readoutFor(props.trace, selection) : undefined;
    return <button type="button" data-trace-route={measured?.route} data-trace-change={measured?.change}
      className={`flow-node ${extra} ${nodeKey(props.selected) === key ? 'node-selected' : ''} ${edited ? 'node-edited' : ''} ${measured ? 'measured-node' : ''}`}
      aria-label={selection.kind === 'head' ? `Inspect Layer ${selection.layer + 1} Head ${selection.head + 1}` : selection.kind === 'stage' ? `Inspect Layer ${selection.layer + 1} ${title}` : `Inspect ${title}`}
      aria-pressed={nodeKey(props.selected) === key} disabled={props.disabled} onClick={() => props.onSelect(selection)}>
      <strong>{title}</strong>{detail && <span>{detail}</span>}{edited && <span className="node-tag">{props.trace && props.intervention?.kind === 'head.scale' && props.intervention.tokens.kind === 'indices' ? `Edited at position ${props.intervention.tokens.indices.join(', ')}` : 'Edited here'}</span>}
      {measured && <>
        <span className="trace-node-position">{selection.kind === 'readout' ? `Prediction position ${props.trace!.report.predictionPosition}` : `At position ${props.trace!.position}`}</span>
        {measured.route !== 'unaffected' && <span className={`route-tag route-${measured.route}`}>{routeLabel[measured.route]}</span>}
        <span className={`change-tag change-${measured.change}`} title={`L2 difference: ${traceNumber(measured.differenceNorm)}`}>
          {selection.kind === 'head' ? 'Output: ' : ''}{changeLabel[measured.change]}{measured.maxDifference !== 0 ? ` · Δ ${traceNumber(measured.maxDifference)}` : ''}</span>
      </>}
      {weights && <span className={`change-tag change-${weights.change}`}>Attention: {changeLabel[weights.change]}{weights.maxDifference !== 0 ? ` · Δ ${traceNumber(weights.maxDifference)}` : ''}</span>}
      {readout && props.symbols && <span className="node-readout" title={`Probability of ${props.symbols[props.target ?? 0]}: ${(readoutProbabilities(readout.baselineLogits, props.target ?? 0).targetProbability * 100).toFixed(4)}% → ${(readoutProbabilities(readout.modifiedLogits, props.target ?? 0).targetProbability * 100).toFixed(4)}%`}>
        {readout.prediction ? 'Prediction' : 'Diagnostic'} @{props.trace!.report.predictionPosition}: {props.symbols[readoutProbabilities(readout.baselineLogits, 0).top]} → {props.symbols[readoutProbabilities(readout.modifiedLogits, 0).top]}
      </span>}
    </button>;
  }
  const arrow = (from: FlowNode[], to: FlowNode) => {
    const route = props.trace && measurementFor(props.trace, to)?.route;
    const connected = props.trace && from.some(item => measurementFor(props.trace!, item)?.route === route);
    return <span className={`flow-arrow ${connected && route !== 'unaffected' ? `trace-arrow-${route}` : ''}`} aria-hidden="true">↓</span>;
  };
  const connection = props.trace ? handoff(props.trace.report, props.trace.origin) : null;
  return <div className={`computation-graph ${props.trace ? 'traced-graph' : ''}`} aria-label="Transformer computation graph">
    {node({ kind: 'embedding' }, 'Input embeddings', 'Token + learned position')}{arrow([{ kind: 'embedding' }], { kind: 'stage', layer: 0, stage: 'block.input' })}
    <div className="flow-layers" style={{ gridTemplateColumns: `repeat(${props.config.layers}, minmax(0, 1fr))` }}>
      {Array.from({ length: props.config.layers }, (_, layer) => <section className="flow-layer" key={layer} aria-label={`Layer ${layer + 1} computation`}>
        <button type="button" className="layer-toggle" aria-expanded={!collapsed.includes(layer)} aria-controls={`flow-layer-${layer}`} onClick={() => setCollapsed(previous => previous.includes(layer) ? previous.filter(item => item !== layer) : [...previous, layer])}>
          <span><span className="layer-number">{String(layer + 1).padStart(2, '0')}</span><strong>Layer {layer + 1}</strong></span><span aria-hidden="true">{collapsed.includes(layer) ? '+' : '−'}</span>
        </button>
        <div id={`flow-layer-${layer}`} hidden={collapsed.includes(layer)}>
          <div className={`residual-branch attention-branch ${props.trace && measurementFor(props.trace, { kind: 'stage', layer, stage: 'block.input' })?.route === 'prediction' ? 'trace-bypass' : ''}`}>
            {node({ kind: 'stage', layer, stage: 'block.input' }, 'Layer input')}{arrow([{ kind: 'stage', layer, stage: 'block.input' }], { kind: 'stage', layer, stage: 'attention.normalized' })}
            {node({ kind: 'stage', layer, stage: 'attention.normalized' }, 'Normalize for attention')}{arrow([{ kind: 'stage', layer, stage: 'attention.normalized' }], { kind: 'head', layer, head: 0 })}
            {connection?.layer === layer && <div className="trace-handoff" data-testid="trace-handoff">Position {connection.from} → {connection.to}<small>{connection.allPositions ? 'One route among all edited positions' : 'Possible route through this attention layer'}</small></div>}
            <div className="flow-heads" aria-label={`Layer ${layer + 1} parallel attention heads`}>
              {Array.from({ length: props.config.heads }, (_, head) => <div key={head}>{node({ kind: 'head', layer, head }, `Head ${head + 1}`, 'Attention → output', 'head-node')}</div>)}
            </div>{arrow(Array.from({ length: props.config.heads }, (_, head) => ({ kind: 'head', layer, head })), { kind: 'stage', layer, stage: 'attention.projected' })}
            {node({ kind: 'stage', layer, stage: 'attention.projected' }, 'Join heads + project')}{arrow([{ kind: 'stage', layer, stage: 'attention.projected' }], { kind: 'stage', layer, stage: 'attention.residual' })}
            {node({ kind: 'stage', layer, stage: 'attention.residual' }, 'Residual after attention', 'Add the layer input', 'add-node')}
            <span className="bypass-label">residual</span>
          </div>
          <div className={`residual-branch mlp-branch ${props.trace && measurementFor(props.trace, { kind: 'stage', layer, stage: 'attention.residual' })?.route === 'prediction' ? 'trace-bypass' : ''}`}>{arrow([{ kind: 'stage', layer, stage: 'attention.residual' }], { kind: 'stage', layer, stage: 'mlp.output' })}
            {node({ kind: 'stage', layer, stage: 'mlp.output' }, 'Normalize + MLP', 'Linear → GELU → linear')}{arrow([{ kind: 'stage', layer, stage: 'mlp.output' }], { kind: 'stage', layer, stage: 'block.output' })}
            {node({ kind: 'stage', layer, stage: 'block.output' }, 'Layer output', 'Add the MLP contribution', 'add-node')}
            <span className="bypass-label">residual</span>
          </div>
        </div>
        {collapsed.includes(layer) && <p className="collapsed-layer">{props.config.heads} parallel heads · MLP · residual additions{props.trace && <span>Trace at position {props.trace.position} · {props.trace.report.nodes.some(item => item.layer === layer && item.position === props.trace!.position && item.route === 'prediction') ? 'Possible prediction route' : 'Outside prediction route'}</span>}</p>}
      </section>)}
    </div>{arrow([{ kind: 'stage', layer: props.config.layers - 1, stage: 'block.output' }], { kind: 'readout' })}
    {node({ kind: 'readout' }, 'Readout → prediction', 'Final normalization · tied unembedding · softmax')}
    <div className="graph-legend"><span className="selection-key">Selected for inspection</span><span className="edit-key">Completed intervention</span></div>
    {props.trace && <p className="trace-legend">Solid route: can affect this prediction. Dashed local route: cannot reach it. Exact match, small difference, and changed badges describe measured vectors.</p>}
  </div>;
}
