import { useEffect, useRef, useState } from 'react';
import type { StorySelection } from '../continuation';
import type { HeadScaleIntervention, ModelManifest } from '../engine/contracts';
import type { BpeDocument } from '../engine/bpe';
import { tokenLabel } from '../engine/tokenizer';
import { scanStoryHeads, rankHeadScan, type HeadScanRow, type ScanOrder } from '../story-head-scan';
import { ModelClient } from '../workers/client';
import { loadStoryModel, predictStory } from '../workers/story-runtime';
interface Props { source: StorySelection; manifest: ModelManifest; tokenizer: BpeDocument; disabled: boolean; onStart: () => void; onChoose: (edit: HeadScaleIntervention) => void; onBusy: (busy: boolean) => void }
export function StoryHeadScan(props: Props) {
  const [rows, setRows] = useState<HeadScanRow[]>([]), [busy, setBusy] = useState(false), [status, setStatus] = useState('Ready to scan.'), [error, setError] = useState('');
  const [order, setOrder] = useState<ScanOrder>('absolute'), [all, setAll] = useState(false);
  const worker = useRef<ModelClient | null>(null), epoch = useRef(0);
  const total = props.manifest.config.layers * props.manifest.config.heads * 3;
  useEffect(() => () => { ++epoch.current; worker.current?.dispose(); }, []);
  function stop() { ++epoch.current; worker.current?.dispose(); worker.current = null; setBusy(false); props.onBusy(false); setStatus('Scan stopped. Results below are partial.'); }
  async function scan() {
    if (busy || props.disabled) return;
    props.onStart(); setRows([]); setAll(false); setError(''); setBusy(true); props.onBusy(true); setStatus('Scanning…');
    const generation = ++epoch.current, current = () => generation === epoch.current;
    const owner = new ModelClient(); worker.current = owner;
    try {
      await loadStoryModel(owner, props.manifest);
      if (!current()) return;
      await scanStoryHeads(props.source, props.manifest.config, (ids, edits) => predictStory(owner, ids, edits), row => { if (current()) setRows(old => [...old, row]); }, () => !current());
      if (current()) setStatus('Scan complete.');
    } catch(e) { if (current()) { setError(String(e)); setStatus('Scan failed. Results below are partial.'); } }
    finally { owner.dispose(); if (current()) { worker.current = null; setBusy(false); props.onBusy(false); } }
  }
  const sorted = rankHeadScan(rows, order), visible = all ? sorted : sorted.slice(0, 8);
  return <section className="story-head-scan" aria-label="Scan heads at selected token">
    <p className="eyebrow">Find a change worth exploring</p><h3>Which heads affect this token most?</h3>
    <p>Scan Mute (0×), Normal (1×), and Amplify (2×) for every head at story step {props.source.step}. Each trial uses the same text and captured settings, changing one head at all input positions.</p>
    <div className="continuation-actions"><button type="button" disabled={busy || props.disabled} onClick={() => void scan()}>Scan all heads</button><button type="button" disabled={!busy} onClick={stop}>Stop head scan</button>
      <label>Sort results <select aria-label="Sort head scan results" value={order} onChange={e => setOrder(e.target.value as ScanOrder)}><option value="absolute">Largest absolute change</option><option value="increase">Largest increase</option><option value="decrease">Largest decrease</option></select></label></div>
    <p role="status" data-testid="head-scan-status">{status} {rows.length} / {total} changes measured.</p>
    {busy && <progress value={rows.length} max={total} aria-label="Head scan progress" />}
    {error && <p role="alert" className="error">Could not finish the scan: {error}</p>}
    {!!rows.length && <><p>Selected token: <strong>{tokenLabel(props.tokenizer, props.source.choice.tokenId)}</strong> · Source probability {(rows[0].before * 100).toFixed(4)}%.</p>
      <ol className="head-scan-results" aria-label="Ranked head changes">{visible.map(row => <li key={`${row.edit.layer}:${row.edit.head}:${row.edit.factor}`} data-scan-layer={row.edit.layer} data-scan-head={row.edit.head} data-scan-factor={row.edit.factor}>
        <div><strong>Layer {row.edit.layer+1} · Head {row.edit.head+1} · {row.edit.factor === 0 ? 'Mute' : row.edit.factor === 1 ? 'Normal' : 'Amplify'}</strong><p><span data-testid="scan-after">{(row.after*100).toFixed(4)}%</span> · <b data-testid="scan-delta">{row.delta > 0 ? '+' : ''}{(row.delta*100).toFixed(4)} percentage points</b> · rank {row.rank}</p><small>{row.choice === props.source.choice.tokenId ? 'Same next token' : `Different next token: ${tokenLabel(props.tokenizer, row.choice)}`}</small></div>
        <button type="button" disabled={busy || props.disabled} onClick={() => props.onChoose(row.edit)} aria-label={`Use layer ${row.edit.layer+1} head ${row.edit.head+1} strength ${row.edit.factor}`}>Use this change</button>
      </li>)}</ol>
      {rows.length > 8 && <button type="button" onClick={() => setAll(!all)}>{all ? 'Show strongest eight' : `Show all ${rows.length} results`}</button>}
      <p className="hint">Raw model probability changes before sampling. These are individual trials relative to the source; head effects can interact and need not add up. Partial results cannot establish the strongest change among unmeasured heads. Normal may restore a previously changed head. Choosing a result fills the draft below; choose its duration and token limit, then Compare from this token.</p></>}
  </section>;
}
