import type { ComparisonResult } from '../engine/runs';
import type { ModelConfig } from '../engine/contracts';
import { nodeKey, type FlowNode } from '../flow';

interface Props {
  config: ModelConfig; tokens: string[]; selected: FlowNode; comparison: ComparisonResult;
  attention: number[]; layer: number; head: number; position: number; pending: boolean; ended: boolean; disabled: boolean;
  onSelect: (node: FlowNode) => void; onDetails?: () => void;
  headControl?: { factor: number; scope: string; onChange: (factor: number) => void };
  onReset?: () => void;
  history?: { edited: boolean };
}
const explanation = (node: FlowNode) => node.kind === 'embedding'
  ? ['Turn symbols into numbers', 'Each symbol gets a learned list of numbers. The model adds its position so that order matters.']
  : node.kind === 'head' ? ['Let symbols share information', 'An attention head gathers information from earlier symbols. Select a head, then use Mute, Normal, or Amplify to change how much its output contributes.']
  : node.kind === 'readout' ? ['Make a prediction', 'The final numbers become a probability for every possible next symbol. Compare the original answer with the answer after your change.']
  : node.stage === 'mlp.output' ? ['Process what was gathered', 'A small neural network transforms each position’s numbers. Its result is added to the running information passed through the model.']
  : ['Carry information forward', 'Each layer adds its attention and processing results to the information it received. These shortcut connections are called residual connections.'];

export function VisualFlow(props: Props) {
  const [title, description] = explanation(props.selected);
  const historical = !!props.history, editedHistory = props.history?.edited;
  const originalLabel = historical ? 'Without edit · same context' : 'Original';
  const changedLabel = historical ? 'Captured edit · same context' : 'After change';
  const active = props.comparison.interventions[0];
  const compact = props.config.layers > 2;
  const layer = historical ? props.layer : props.selected.kind === 'head' || props.selected.kind === 'stage' ? props.selected.layer : active?.layer ?? 0;
  const visibleLayers = compact ? [layer] : Array.from({ length: props.config.layers }, (_, i) => i);
  const selectedHead = historical ? props.head : props.selected.kind === 'head' ? props.selected.head : 0;
  const rows = [...props.comparison.displayTokens].sort((a, b) => b.probability - a.probability || a.tokenId - b.tokenId).slice(0, 4);
  const query = Math.min(props.position, props.tokens.length - 1);
  const weights = props.attention.slice(query * props.tokens.length, (query + 1) * props.tokens.length);
  const sources = weights.map((weight, position) => ({ weight, position })).filter(row => row.position <= query)
    .sort((a, b) => b.weight - a.weight || a.position - b.position).slice(0, 4).sort((a, b) => a.position - b.position);
  function select(node: FlowNode, label: string, detail: string, className = '') {
    const applied = node.kind === 'head' ? props.comparison.interventions.find(edit => edit.kind !== 'residual.replace' && edit.layer === node.layer && edit.head === node.head) : undefined;
    const edited = applied && (applied.kind !== 'head.scale' || applied.factor !== 1);
    return <button type="button" className={`visual-node ${className} ${edited ? 'visual-edited' : ''}`} disabled={props.disabled}
      aria-label={label} aria-pressed={nodeKey(node) === nodeKey(props.selected)} onClick={() => props.onSelect(node)}>
      <strong>{label}</strong><span>{detail}</span>{edited && <small>{historical ? 'Captured edit' : 'Your change'}{applied.kind === 'head.scale' ? ` · ${applied.factor.toFixed(2)}×` : ' · attention redirected'}</small>}
    </button>;
  }
  return <section className={`visual-flow ${props.pending || props.ended ? 'visual-pending' : ''}`} aria-label="Visual transformer flow" aria-busy={props.pending}>
    <div className="visual-intro"><div><p className="eyebrow">From symbols to a prediction</p><h2>Follow the information.</h2></div>
      <p>{historical ? 'Click a stage to inspect this recorded step. The context and edit stay fixed.' : 'Click a stage to explore it. Change a head and see what happens to the answer.'}</p></div>
    {compact && <div className="visual-layer-navigation" role="group" aria-label="Model layer navigation">
      <button type="button" aria-label="Previous model layer" disabled={props.disabled || layer === 0} onClick={() => props.onSelect({ kind: 'head', layer: layer - 1, head: selectedHead })}>←</button>
      <strong>Layer {layer + 1} of {props.config.layers}</strong>
      <button type="button" aria-label="Next model layer" disabled={props.disabled || layer === props.config.layers - 1} onClick={() => props.onSelect({ kind: 'head', layer: layer + 1, head: selectedHead })}>→</button>
      <span>All {props.config.layers} layers run. Choose one to inspect.</span>
    </div>}
    <div className="visual-scroll" tabIndex={0} role="region" aria-label="Scrollable model flow">
      <div className="visual-canvas" style={{ gridTemplateColumns: `170px repeat(${visibleLayers.length}, minmax(260px, 1fr)) 205px` }}>
        <div className="visual-embedding"><p className="visual-stage-label">01 <strong>Embedding</strong></p><p className="visual-subtitle">Give each symbol a place</p>
          <div className="visual-token-stream">{props.tokens.map((token, i) => <span key={i} className={i === props.position ? 'visual-token-current' : ''} title={`Position ${i}: ${token}`}>{token}</span>)}</div>
          <svg className="embedding-ribbons" viewBox="0 0 160 110" aria-hidden="true">{['#528acf', '#cf7096', '#218a89', '#7963cb'].map((color, i) => <path key={color} d={`M 12 ${15 + i * 22} C 75 ${15 + i * 22}, 85 55, 152 55`} fill="none" stroke={color} strokeWidth="10" opacity=".22" />)}</svg>
          {select({ kind: 'embedding' }, 'Read the input', 'Symbols + their positions')}
        </div>
        {visibleLayers.map(layer => <div className="visual-block" key={layer}>
          <svg className="visual-layer-link" viewBox="0 0 32 100" preserveAspectRatio="none" aria-hidden="true"><path d="M 0 100 C 36 100, -4 0, 32 0" /></svg>
          <p className="visual-stage-label">{String(layer + 2).padStart(2, '0')} <strong>Transformer layer {layer + 1}</strong></p>
          <p className="visual-subtitle">{compact && layer > 0 ? `${layer} earlier ${layer === 1 ? 'layer has' : 'layers have'} already processed the input` : 'Share information, then process it'}</p>
          <div className="visual-residual"><span>Carry the input forward</span><i aria-hidden="true">→</i></div>
          <p className="visual-part-label">Attention <span>{props.config.heads} heads working together</span></p>
          <svg className="head-ribbons" viewBox="0 0 240 44" preserveAspectRatio="none" aria-hidden="true">{Array.from({ length: props.config.heads }, (_, head) => <path key={head} d={`M 120 0 C 120 20, ${(head + .5) * 240 / props.config.heads} 20, ${(head + .5) * 240 / props.config.heads} 44`} />)}</svg>
          <div className="visual-heads" style={{ gridTemplateColumns: `repeat(${props.config.heads}, minmax(0, 1fr))` }}>{Array.from({ length: props.config.heads }, (_, head) => <div key={head}>{select({ kind: 'head', layer, head }, `L${layer + 1} · Head ${head + 1}`, 'Gather', 'visual-head')}</div>)}</div>
          <svg className="head-ribbons ribbon-join" viewBox="0 0 240 44" preserveAspectRatio="none" aria-hidden="true">{Array.from({ length: props.config.heads }, (_, head) => <path key={head} d={`M ${(head + .5) * 240 / props.config.heads} 0 C ${(head + .5) * 240 / props.config.heads} 24, 120 24, 120 44`} />)}</svg>
          {select({ kind: 'stage', layer, stage: 'attention.residual' }, `Combine · layer ${layer + 1}`, 'Join heads + add the input', 'visual-combine')}
          <span className="visual-down" aria-hidden="true">↓</span>
          {select({ kind: 'stage', layer, stage: 'mlp.output' }, `Process · layer ${layer + 1}`, 'Transform each position’s numbers', 'visual-mlp')}
          <span className="visual-down" aria-hidden="true">↓</span>
          {select({ kind: 'stage', layer, stage: 'block.output' }, `Pass it on · layer ${layer + 1}`, 'Add the processing result', 'visual-output')}
        </div>)}
        <div className="visual-predictions"><p className="visual-stage-label">{String(props.config.layers + 2).padStart(2, '0')} <strong>Prediction</strong></p><p className="visual-subtitle">{compact && layer < props.config.layers - 1 ? `After ${props.config.layers - layer - 1} more ${props.config.layers - layer - 1 === 1 ? 'layer' : 'layers'} · what comes next?` : 'What comes next?'}</p>
          <div className="visual-answer"><span>{historical ? 'Most likely token' : props.pending || props.ended ? 'Last completed answer' : 'After your change'}</span><strong>{props.comparison.modifiedTopTokens[0].text}</strong>{(!historical || editedHistory) && <small>{historical ? 'Without edit, same context' : 'Original answer'}: {props.comparison.baselineTopTokens[0].text}</small>}</div>
          <div className="visual-probabilities" aria-label="Flow prediction probabilities">{rows.map(row => <div key={row.tokenId}><span>{row.text}</span><div><i style={{ width: `${row.baselineProbability * 100}%` }} /><b style={{ width: `${row.probability * 100}%` }} /></div><strong>{(row.probability * 100).toFixed(1)}%</strong></div>)}</div>
          <p className="visual-probability-key">{historical && !editedHistory ? <><b /> Original model</> : <><i /> {originalLabel} <b /> {changedLabel}</>}</p>
          {props.onReset && <button type="button" className="visual-reset" disabled={props.disabled} onClick={props.onReset}>Restore original prediction</button>}
          {select({ kind: 'readout' }, 'Read the prediction', 'All symbols get a probability')}
        </div>
      </div>
    </div>
    {props.selected.kind === 'head' && props.headControl && <div className="visual-head-control" role="group" aria-label="Change the selected head">
      <div className="visual-control-title"><p className="eyebrow">Try a change</p><h3>Layer {props.selected.layer + 1} · Head {props.selected.head + 1}</h3><p>{props.headControl.scope}</p></div>
      <div className="visual-presets">{[{ factor: 0, label: 'Mute', detail: 'Remove its output' }, { factor: 1, label: 'Normal', detail: 'Use its original output' }, { factor: 2, label: 'Amplify', detail: 'Double its output' }].map(preset =>
        <button type="button" key={preset.factor} aria-label={`${preset.label} selected head`} aria-pressed={props.headControl!.factor === preset.factor}
          disabled={props.disabled} onClick={() => props.headControl!.onChange(preset.factor)}><strong>{preset.label}</strong><span>{preset.detail}</span></button>)}</div>
      <label className="visual-strength">Output strength <output>{props.headControl.factor.toFixed(2)}×</output><input type="range" min="0" max="2" step="any"
        aria-label="Selected head output strength" value={props.headControl.factor} disabled={props.disabled} onChange={event => props.headControl!.onChange(Number(event.target.value))} /></label>
      <p className="visual-control-note">{props.ended ? 'Session ended. This is the last completed prediction.' : props.pending ? 'Updating… the diagram still shows the last completed prediction.' : 'The diagram shows the completed prediction.'} Attention weights stay the same when you scale the output.</p>
    </div>}
    <div className="visual-explanation"><span className="visual-explanation-icon" aria-hidden="true">↳</span><div><h3>{title}</h3><p>{historical && props.selected.kind === 'head' ? 'This attention head gathers information from earlier tokens. Its measured output at the final input position appears below.' : historical && props.selected.kind === 'readout' ? 'These are the model’s next-token probabilities before sampling. The selected generated token is shown above; it need not be the most likely one.' : description}</p></div>{props.onDetails && <button type="button" className="text-button" onClick={props.onDetails}>Open detailed computation →</button>}</div>
    {sources.length > 0 && <details className="visual-attention"><summary>{historical ? 'Where did this head look?' : 'Where did the edited head look?'} <span>Layer {props.layer + 1} · Head {props.head + 1}</span></summary>
      <div className="attention-story"><svg viewBox="0 0 420 150" role="img" aria-label={`Largest attention weights for query position ${query}`}>
        {sources.map(({ weight, position }, i) => <g key={position}><path d={`M 135 ${24 + i * 34} C 220 ${24 + i * 34}, 220 75, 290 75`} fill="none" stroke="#7963cb" strokeWidth={1 + weight * 16} opacity={.25 + weight * .65} /><text x="4" y={28 + i * 34}>{props.tokens[position]} · pos {position}</text><text x="105" y={28 + i * 34} textAnchor="end">{(weight * 100).toFixed(1)}%</text></g>)}<rect x="290" y="54" width="118" height="42" rx="8" fill="#eee9fb" /><text x="349" y="80" textAnchor="middle">{props.tokens[query]} · pos {query}</text>
      </svg><p>These are the four largest measured attention weights for this query. Thicker lines mean more attention. {historical ? 'Weights come from the selected story branch. An edit in an earlier layer can change later attention.' : active?.kind === 'attention.override' ? 'This experiment directly replaces the attention weights for one query.' : 'Scaling a head changes its output, while its attention weights stay the same.'}</p></div>
    </details>}
    <p className="visual-caption">The main connections show computation order, not measured strength. {historical ? 'Normalization is included within each stage. Select a stage to inspect its values below.' : 'Normalization is included within each stage; Detailed view shows every step.'}</p>
  </section>;
}
