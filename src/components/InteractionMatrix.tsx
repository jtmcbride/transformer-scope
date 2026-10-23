import { useRef, useState, type KeyboardEvent } from 'react';
import type { ModelConfig } from '../engine/contracts';
import { interactionFill, interactionInk } from '../head-pair-heatmap';
export interface InteractionCell { key: string; value: number; description?: string }
interface Props { cells: (InteractionCell | undefined)[][]; config: ModelConfig; rowFactor: number; columnFactor: number; limit: number; selectedKey: string; disabled: boolean; onSelect: (key: string) => void; valueLabel?: string }
const points = (n: number) => `${n > 0 ? '+' : ''}${(n*100).toFixed(4)} percentage points`;
export function InteractionMatrix({ cells, config, rowFactor, columnFactor, limit, selectedKey, disabled, onSelect, valueLabel = 'Interaction' }: Props) {
  const [focus,setFocus] = useState('');
  const count = config.layers*config.heads, table = useRef<HTMLTableElement>(null);
  const first = cells.flatMap((line,r) => line.flatMap((cell,c) => cell ? [`${r}:${c}`] : []))[0];
  const [fr,fc] = focus.split(':').map(Number);
  const tabStop = cells[fr]?.[fc] ? focus : first;
  function move(event: KeyboardEvent<HTMLButtonElement>, r: number, c: number) {
    if (event.altKey || event.metaKey) return;
    const deltas: Record<string, [number,number]> = { ArrowUp: [-1,0], ArrowDown: [1,0], ArrowLeft: [0,-1], ArrowRight: [0,1] };
    let target: [number,number] | undefined;
    if (event.key === 'Home' || event.key === 'End') {
      const candidates = cells.flatMap((line,i) => line.flatMap((cell,j) => cell && (event.ctrlKey || i === r) ? [[i,j] as [number,number]] : []));
      target = event.key === 'Home' ? candidates[0] : candidates.at(-1);
    } else if (deltas[event.key]) {
      const [dr,dc] = deltas[event.key];
      for (let i = r+dr, j = c+dc; i >= 0 && i < count && j >= 0 && j < count; i += dr, j += dc) if (cells[i][j]) { target = [i,j]; break; }
    } else return;
    event.preventDefault();
    if (target) table.current?.querySelector<HTMLButtonElement>(`button[data-map-row="${target[0]}"][data-map-column="${target[1]}"]`)?.focus();
  }
  return (
<div className="pair-map-scroll" role="region" aria-label="Scrollable head-pair map" tabIndex={0}>
        <table ref={table} className="pair-map-table"><caption>Rows: {rowFactor}× strength · Columns: {columnFactor}× strength</caption>
          <thead><tr><th rowSpan={2} scope="col">Head ↓ / →</th>{Array.from({ length: config.layers }, (_,i) => <th key={i} scope="colgroup" colSpan={config.heads}>Layer {i+1}</th>)}</tr>
            <tr>{Array.from({ length: count }, (_,i) => <th key={i} scope="col" className={i%config.heads === 0 ? 'pair-map-layer-start' : ''}>H{i%config.heads+1}</th>)}</tr></thead>
          <tbody>{cells.map((line,r) => <tr key={r} className={r%config.heads === 0 ? 'pair-map-layer-row' : ''}><th scope="row">L{Math.floor(r/config.heads)+1} · H{r%config.heads+1}</th>{line.map((cell,c) => {
            const label = `Row layer ${Math.floor(r/config.heads)+1} head ${r%config.heads+1} strength ${rowFactor}; column layer ${Math.floor(c/config.heads)+1} head ${c%config.heads+1} strength ${columnFactor}`;
            const key = `${r}:${c}`;
            return <td key={c} className={c%config.heads === 0 ? 'pair-map-layer-start' : ''}>{cell ? <button type="button" className="pair-map-cell" disabled={disabled}
              data-map-row={r} data-map-column={c} data-interaction={cell.value} data-pair-key={cell.key}
              aria-label={`${label}. ${valueLabel} ${points(cell.value)}. ${cell.description ?? ''}`} aria-pressed={cell.key === selectedKey}
              title={`${label}\n${valueLabel}: ${points(cell.value)}\n${cell.description ?? ''}`}
              tabIndex={key === tabStop ? 0 : -1} onFocus={() => setFocus(key)} onKeyDown={event => move(event,r,c)} onClick={() => onSelect(cell.key)}
              style={{ backgroundColor: interactionFill(cell.value,limit), color: interactionInk(cell.value,limit) }}><span aria-hidden="true">{cell.value > 0 ? '+' : cell.value < 0 ? '−' : '0'}</span></button>
              : <span className={r === c ? 'pair-map-diagonal' : 'pair-map-pending'} title={`${label}. ${r === c ? 'Same head: not a pair' : 'Not measured'}`} aria-label={`${label}. ${r === c ? 'Same head: not a pair' : 'Not measured'}`}>{r === c ? '×' : '·'}</span>}</td>;
          })}</tr>)}</tbody>
        </table>
      </div>
  );
}
