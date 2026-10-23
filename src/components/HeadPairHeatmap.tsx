import { useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { ModelConfig } from '../engine/contracts';
import { headPairHeatmap, headPairKey, interactionFill, interactionInk } from '../head-pair-heatmap';
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
  const [assignment, setAssignment] = useState(''), [focus, setFocus] = useState('');
  const [rowFactor, columnFactor] = options.find(pair => pair.join(':') === assignment) ?? options[0];
  const map = useMemo(() => headPairHeatmap(rows,config,rowFactor,columnFactor), [rows,config,rowFactor,columnFactor]);
  const selected = rows.find(row => headPairKey(row.edits) === selectedKey);
  const selectedVisible = selected && map.cells.some(line => line.includes(selected));
  const count = config.layers*config.heads, table = useRef<HTMLTableElement>(null);
  const first = map.cells.flatMap((line,r) => line.flatMap((cell,c) => cell ? [`${r}:${c}`] : []))[0];
  const [fr,fc] = focus.split(':').map(Number);
  const tabStop = map.cells[fr]?.[fc] ? focus : first;
  function move(event: KeyboardEvent<HTMLButtonElement>, r: number, c: number) {
    if (event.altKey || event.metaKey) return;
    const deltas: Record<string, [number,number]> = { ArrowUp: [-1,0], ArrowDown: [1,0], ArrowLeft: [0,-1], ArrowRight: [0,1] };
    let target: [number,number] | undefined;
    if (event.key === 'Home' || event.key === 'End') {
      const cells = map.cells.flatMap((line,i) => line.flatMap((cell,j) => cell && (event.ctrlKey || i === r) ? [[i,j] as [number,number]] : []));
      target = event.key === 'Home' ? cells[0] : cells.at(-1);
    } else if (deltas[event.key]) {
      const [dr,dc] = deltas[event.key];
      for (let i = r+dr, j = c+dc; i >= 0 && i < count && j >= 0 && j < count; i += dr, j += dc) if (map.cells[i][j]) { target = [i,j]; break; }
    } else return;
    event.preventDefault();
    if (target) table.current?.querySelector<HTMLButtonElement>(`button[data-map-row="${target[0]}"][data-map-column="${target[1]}"]`)?.focus();
  }
  return <section className="head-pair-map" aria-label="Head-pair interaction heatmap">
    <div className="pair-map-heading"><div><p className="eyebrow">See the pattern across heads</p><h4>Interaction map</h4></div>
      {options.length > 1 && <label>Map strengths<select aria-label="Heatmap strength assignment" value={`${rowFactor}:${columnFactor}`} onChange={event => setAssignment(event.target.value)}>{options.map(([a,b]) => <option key={`${a}:${b}`} value={`${a}:${b}`}>Rows {a}× · Columns {b}×</option>)}</select></label>}
    </div>
    <p>Read one head down the side and another across the top. After the scan completes or stops, select a measured cell to see the pair, then preview it and branch the story.</p>
    <div className="pair-map-layout"><div className="pair-map-plot">
      <div className="pair-map-scroll" role="region" aria-label="Scrollable head-pair map" tabIndex={0}>
        <table ref={table} className="pair-map-table"><caption>Rows: {rowFactor}× strength · Columns: {columnFactor}× strength</caption>
          <thead><tr><th rowSpan={2} scope="col">Head ↓ / →</th>{Array.from({ length: config.layers }, (_,i) => <th key={i} scope="colgroup" colSpan={config.heads}>Layer {i+1}</th>)}</tr>
            <tr>{Array.from({ length: count }, (_,i) => <th key={i} scope="col" className={i%config.heads === 0 ? 'pair-map-layer-start' : ''}>H{i%config.heads+1}</th>)}</tr></thead>
          <tbody>{map.cells.map((line,r) => <tr key={r} className={r%config.heads === 0 ? 'pair-map-layer-row' : ''}><th scope="row">L{Math.floor(r/config.heads)+1} · H{r%config.heads+1}</th>{line.map((cell,c) => {
            const label = `Row layer ${Math.floor(r/config.heads)+1} head ${r%config.heads+1} strength ${rowFactor}; column layer ${Math.floor(c/config.heads)+1} head ${c%config.heads+1} strength ${columnFactor}`;
            const key = `${r}:${c}`;
            return <td key={c} className={c%config.heads === 0 ? 'pair-map-layer-start' : ''}>{cell ? <button type="button" className="pair-map-cell" disabled={disabled}
              data-map-row={r} data-map-column={c} data-interaction={cell.interaction} data-pair-key={headPairKey(cell.edits)}
              aria-label={`${label}. Interaction ${points(cell.interaction)}. Combined change ${points(cell.delta)}.`} aria-pressed={headPairKey(cell.edits) === selectedKey}
              title={`${label}\nInteraction: ${points(cell.interaction)}\nCombined change: ${points(cell.delta)}`}
              tabIndex={key === tabStop ? 0 : -1} onFocus={() => setFocus(key)} onKeyDown={event => move(event,r,c)} onClick={() => onSelect(cell)}
              style={{ backgroundColor: interactionFill(cell.interaction,map.limit), color: interactionInk(cell.interaction,map.limit) }}><span aria-hidden="true">{cell.interaction > 0 ? '+' : cell.interaction < 0 ? '−' : '0'}</span></button>
              : <span className={r === c ? 'pair-map-diagonal' : 'pair-map-pending'} title={`${label}. ${r === c ? 'Same head: not a pair' : 'Not measured'}`} aria-label={`${label}. ${r === c ? 'Same head: not a pair' : 'Not measured'}`}>{r === c ? '×' : '·'}</span>}</td>;
          })}</tr>)}</tbody>
        </table>
      </div>
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
