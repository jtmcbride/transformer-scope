import { useEffect, useRef, useState } from 'react';
import type { StoryScanProps } from './StoryHeadScan';
import type { ScanOrder } from '../story-head-scan';
import { pairStrengthOptions, pairScanTotal, rankHeadPairScan, scanStoryHeadPairs, type PairStrengths, type HeadPairScanRow } from '../story-head-pair-scan';
import { tokenLabel } from '../engine/tokenizer';
import { ModelClient } from '../workers/client';
import { loadStoryModel, withStoryPrefix } from '../workers/story-runtime';
const name = (edit: HeadPairScanRow['edits'][number]) => `Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`;
const key = (edit: HeadPairScanRow['edits'][number]) => `${edit.layer}:${edit.head}:${edit.factor}`;
export function StoryHeadPairScan(props: StoryScanProps) {
  const [strengths, setStrengths] = useState<PairStrengths>('mute'), [order, setOrder] = useState<ScanOrder>('absolute');
  const [rows, setRows] = useState<HeadPairScanRow[]>([]), [all, setAll] = useState(false);
  const [busy, setBusy] = useState(false), [status, setStatus] = useState('Ready to scan.'), [error, setError] = useState('');
  const worker = useRef<ModelClient | null>(null), epoch = useRef(0);
  const total = pairScanTotal(props.manifest.config, strengths);
  useEffect(() => () => { ++epoch.current; worker.current?.dispose(); }, []);
  function stop() { ++epoch.current; worker.current?.dispose(); worker.current = null; setBusy(false); props.onBusy(false); setStatus('Pair scan stopped. Results below are partial.'); }
  async function scan() {
    if (busy || props.disabled) return;
    props.onStart(); setRows([]); setAll(false); setError(''); setBusy(true); props.onBusy(true); setStatus('Scanning pairs…');
    const generation = ++epoch.current, current = () => generation === epoch.current;
    const owner = new ModelClient(); worker.current = owner;
    try {
      await loadStoryModel(owner, props.manifest);
      if (!current()) return;
      await withStoryPrefix(owner, props.source.prefix, predict => scanStoryHeadPairs(props.source, props.manifest.config, strengths, predict, row => { if (current()) setRows(old => [...old, row]); }, () => !current()));
      if (current()) setStatus('Pair scan complete.');
    } catch (e) { if (current()) { setError(String(e)); setStatus('Pair scan failed. Results below are partial.'); } }
    finally { owner.dispose(); if (current()) { worker.current = null; setBusy(false); props.onBusy(false); } }
  }
  const sorted = rankHeadPairScan(rows, order), visible = all ? sorted : sorted.slice(0, 8);
  return <section className="story-head-scan story-head-pair-scan" aria-label="Scan head pairs at selected token">
    <p className="eyebrow">Find a combination worth exploring</p><h3>Which pairs affect this token most?</h3>
    <p>Measure two heads together at story step {props.source.step}, using the same starting text and captured source settings. This ranks actual combined probability changes, including interactions.</p>
    <div className="continuation-actions"><label>Pair strengths<select aria-label="Pair scan strengths" value={strengths} disabled={busy || props.disabled} onChange={event => { setStrengths(event.target.value as PairStrengths); setRows([]); setAll(false); setError(''); setStatus('Ready to scan.'); }}>{pairStrengthOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
      <button type="button" disabled={busy || props.disabled} onClick={() => void scan()}>Scan head pairs</button><button type="button" disabled={!busy} onClick={stop}>Stop pair scan</button>
      <label>Sort pairs<select aria-label="Sort pair scan results" value={order} onChange={event => setOrder(event.target.value as ScanOrder)}><option value="absolute">Largest absolute change</option><option value="increase">Largest increase</option><option value="decrease">Largest decrease</option></select></label></div>
    <p className="hint">{total} combinations in this scope. Mixed strengths test both assignments. All strengths test all nine strength assignments per distinct head pair and take longer. Normal may restore a changed head.</p>
    <p role="status" data-testid="pair-scan-status">{status} {rows.length} / {total} combinations measured.</p>
    {busy && <progress value={rows.length} max={total} aria-label="Head pair scan progress" />}
    {error && <p role="alert" className="error">Could not finish the pair scan: {error}</p>}
    {!!rows.length && <><p>Selected token: <strong>{tokenLabel(props.tokenizer, props.source.choice.tokenId)}</strong> · Source probability {(rows[0].before*100).toFixed(4)}%.</p>
      <ol className="head-scan-results" aria-label="Ranked head pair changes">{visible.map(row => <li key={`${key(row.edits[0])}|${key(row.edits[1])}`} data-pair-first={key(row.edits[0])} data-pair-second={key(row.edits[1])}>
        <div><strong>{name(row.edits[0])} + {name(row.edits[1])}</strong><p><span data-testid="pair-scan-after">{(row.after*100).toFixed(4)}%</span> · <b data-testid="pair-scan-delta">{row.delta > 0 ? '+' : ''}{(row.delta*100).toFixed(4)} percentage points</b> · rank {row.rank}</p><small>{row.choice === props.source.choice.tokenId ? 'Same next token' : `Different next token: ${tokenLabel(props.tokenizer, row.choice)}`}</small></div>
        <button type="button" disabled={busy || props.disabled} onClick={() => props.onChoose(row.edits)} aria-label={`Use pair layer ${row.edits[0].layer+1} head ${row.edits[0].head+1} strength ${row.edits[0].factor} and layer ${row.edits[1].layer+1} head ${row.edits[1].head+1} strength ${row.edits[1].factor}`}>Use this pair</button>
      </li>)}</ol>
      {rows.length > 8 && <div className="continuation-actions"><button type="button" disabled={busy} onClick={() => setAll(!all)}>{all ? 'Show strongest eight pairs' : `Show all ${rows.length} pairs`}</button></div>}
      <p className="hint">Raw model probability changes before sampling, relative to the captured source. Rankings cover only the chosen strengths and completed trials; partial results cannot establish the strongest unmeasured pair. Use a pair to preview its individual and combined effects below, choose duration and token limit, then Compare from this token.</p></>}
  </section>;
}
