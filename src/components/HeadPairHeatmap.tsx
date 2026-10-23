import { useMemo, useState } from 'react';
import { InteractionMatrix } from './InteractionMatrix';
import type { ModelConfig } from '../engine/contracts';
import { headPairHeatmap, headPairKey } from '../head-pair-heatmap';
import { pairStrengthOptions, type HeadPairScanRow, type PairStrengths } from '../story-head-pair-scan';
import '../head-pair-heatmap.css';

interface Props {
  rows: HeadPairScanRow[]; config: ModelConfig; strengths: PairStrengths; disabled: boolean;
  selectedKey: string; onSelect: (row: HeadPairScanRow) => void; onChoose: (row: HeadPairScanRow) => void;
}
const points = (n: number) => `${n > 0 ? '+' : ''}${(n*100).toFixed(4)} percentage points`;
const name = (edit: HeadPairScanRow['edits'][number]) => `Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`;
export function HeadPairHeatmap({ rows, config, strengths, disabled, selectedKey, onSelect, onChoose }: Props) {
  const options = pairStrengthOptions.find(option => option.value === strengths)!.factors;
  const [assignment, setAssignment] = useState('');
  const [rowFactor, columnFactor] = options.find(pair => pair.join(':') === assignment) ?? options[0];
  const map = useMemo(() => headPairHeatmap(rows,config,rowFactor,columnFactor), [rows,config,rowFactor,columnFactor]);
  const selected = rows.find(row => headPairKey(row.edits) === selectedKey);
  const selectedVisible = selected && map.cells.some(line => line.includes(selected));
  return <section className="head-pair-map" aria-label="Head-pair interaction heatmap">
    <div className="pair-map-heading"><div><p className="eyebrow">See the pattern across heads</p><h4>Interaction map</h4></div>
      {options.length > 1 && <label>Map strengths<select aria-label="Heatmap strength assignment" value={`${rowFactor}:${columnFactor}`} onChange={event => setAssignment(event.target.value)}>{options.map(([a,b]) => <option key={`${a}:${b}`} value={`${a}:${b}`}>Rows {a}× · Columns {b}×</option>)}</select></label>}
    </div>
    <p>Read one head down the side and another across the top. After the scan completes or stops, select a measured cell to see the pair, then preview it and branch the story.</p>
    <div className="pair-map-layout"><div className="pair-map-plot">
      <InteractionMatrix cells={map.cells.map(line => line.map(cell => cell ? { key: headPairKey(cell.edits), value: cell.interaction, description: `Combined change ${points(cell.delta)}.` } : undefined))} config={config} rowFactor={rowFactor} columnFactor={columnFactor} limit={map.limit} selectedKey={selectedKey} disabled={disabled} onSelect={key => { const row = rows.find(row => headPairKey(row.edits) === key); if (row) onSelect(row); }} />
      <div className="pair-map-legend" aria-label="Interaction color scale">{map.limit > 0 ? <><div className="pair-map-ramp" aria-hidden="true" /><div><span>−{(map.limit*100).toFixed(4)} pp</span><span>0</span><span>+{(map.limit*100).toFixed(4)} pp</span></div></> : <p>{rows.length ? 'All measured interactions are zero.' : 'Color scale will appear with measured results.'}</p>}<p>Blue: negative · Orange: positive · pp = percentage points. Color range uses all measured trials in this scan and grows as results arrive.</p></div>
      <p className="hint" data-testid="heatmap-coverage">{map.measured} / {map.total} unique combinations shown for these axis strengths. {rowFactor === columnFactor ? 'Both halves mirror the same pairs.' : 'Opposite cells swap which head gets each strength.'} The diagonal is excluded. Hatched cells are unmeasured; pale cells marked 0 are measured zeroes.</p>
      <p className="hint">Keyboard: Tab into the map, arrows move between measured cells, Enter selects. Home / End move along a row; Ctrl+Home / Ctrl+End go to the first / last measured cell. Scroll the map to see more heads on small screens.</p>
    </div>
    <div className="pair-map-detail" aria-label="Selected heatmap pair" aria-live="polite">
      {selected ? <><p className="eyebrow">Selected pair</p><h5>{name(selected.edits[0])}<br />{name(selected.edits[1])}</h5>
        {!selectedVisible && <p data-testid="heatmap-outside-view">This selected pair is outside the current axis-strength view. Its measured values and strengths are kept below.</p>}
        <dl><div><dt>Interaction</dt><dd data-testid="heatmap-selected-interaction">{points(selected.interaction)}</dd></div><div><dt>Combined change</dt><dd>{points(selected.delta)}</dd></div><div><dt>Sum of individual changes</dt><dd>{points(selected.sumDelta)}</dd></div>
          {selected.edits.map((edit,i) => <div key={i}><dt>Layer {edit.layer+1}, head {edit.head+1} alone</dt><dd>{points(selected.individualProbabilities[i]-selected.before)}</dd></div>)}<div><dt>Token probability with both</dt><dd>{(selected.after*100).toFixed(4)}%</dd></div></dl>
        <button type="button" disabled={disabled} onClick={() => onChoose(selected)}>Preview and use this pair</button>
        <p className="hint">Opens the probability preview and fills both head changes. Then choose the duration and token limit and compare the stories.</p></> : <><p className="eyebrow">Inspect a combination</p><h5>{map.measured ? 'Choose a colored cell' : rows.length ? 'No measured pairs in this strength view yet' : 'Run a pair scan to fill the map'}</h5><p>Layer blocks help you spot patterns within and between layers. Each colored cell is a measured interaction at this token.</p><p>Select a cell for its exact values. A large interaction can accompany a small combined change.</p></>}
    </div></div>
  </section>;
}
