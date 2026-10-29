import { useEffect, useMemo, useRef, useState } from 'react';
import { HeadPairHeatmap } from './HeadPairHeatmap';
import { headPairKey } from '../head-pair-heatmap';
import type { StoryScanProps } from './StoryHeadScan';
import { pairStrengthOptions, pairScanTotal, rankHeadPairScan, scanStoryHeadPairs, type PairStrengths, type PairScanOrder, type HeadPairScanRow } from '../story-head-pair-scan';
import { tokenLabel } from '../engine/tokenizer';
import { ModelClient } from '../workers/client';
import { loadStoryModel, withStoryPrefix } from '../workers/story-runtime';
const points = (n: number) => `${n > 0 ? '+' : ''}${(n*100).toFixed(4)} percentage points`;
const name = (edit: HeadPairScanRow['edits'][number]) => `Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`;
const key = (edit: HeadPairScanRow['edits'][number]) => `${edit.layer}:${edit.head}:${edit.factor}`;
export function StoryHeadPairScan(props: StoryScanProps & { targetTokenId: number; onTargetChange: (id: number) => void; targeted?: boolean }) {
  const [strengths, setStrengths] = useState<PairStrengths>('mute'), [order, setOrder] = useState<PairScanOrder>(props.targeted ? 'increase' : 'interaction-absolute');
  const [rows, setRows] = useState<HeadPairScanRow[]>([]), [all, setAll] = useState(false);
  const [selectedKey, setSelectedKey] = useState('');
  const [busy, setBusy] = useState(false), [status, setStatus] = useState('Ready to scan.'), [error, setError] = useState('');
  const [started, setStarted] = useState(false);
  const worker = useRef<ModelClient | null>(null), epoch = useRef(0);
  const heading = useRef<HTMLHeadingElement>(null), focused = useRef(false);
  useEffect(() => { if (props.targeted && !props.disabled && !focused.current) { focused.current = true; heading.current?.focus({ preventScroll: true }); heading.current?.scrollIntoView({ block: 'start' }); } }, [props.targeted, props.disabled]);
  const targetOptions = useMemo(() => props.tokenizer.tokens.map((_,id) => id !== 0 && id !== 2 && <option key={id} value={id}>{tokenLabel(props.tokenizer,id)} · token {id}</option>), [props.tokenizer]);
  const total = pairScanTotal(props.manifest.config, strengths);
  useEffect(() => () => { ++epoch.current; worker.current?.dispose(); }, []);
  function stop() { ++epoch.current; worker.current?.dispose(); worker.current = null; setBusy(false); props.onBusy(false); setStatus('Pair scan stopped. Results below are partial.'); }
  useEffect(() => { if (props.active === false && worker.current) stop(); }, [props.active]);
  async function scan() {
    if (busy || props.disabled) return;
    props.onStart(); setStarted(true); setRows([]); setSelectedKey(''); setAll(false); setError(''); setBusy(true); props.onBusy(true); setStatus('Scanning pairs…');
    const generation = ++epoch.current, current = () => generation === epoch.current;
    const owner = new ModelClient(); worker.current = owner;
    try {
      await loadStoryModel(owner, props.manifest);
      if (!current()) return;
      await withStoryPrefix(owner, props.source.prefix, predict => scanStoryHeadPairs(props.source, props.manifest.config, strengths, predict, row => { if (current()) setRows(old => [...old, row]); }, () => !current(), props.targetTokenId));
      if (current()) setStatus('Pair scan complete.');
    } catch (e) { if (current()) { setError(String(e)); setStatus('Pair scan failed. Results below are partial.'); } }
    finally { owner.dispose(); if (current()) { worker.current = null; setBusy(false); props.onBusy(false); } }
  }
  const sorted = useMemo(() => rankHeadPairScan(rows, order), [rows, order]), visible = all ? sorted : sorted.slice(0, 8);
  function choose(row: HeadPairScanRow) { setSelectedKey(headPairKey(row.edits)); props.onChoose(row.edits); }
  return <section className="story-head-scan story-head-pair-scan" aria-label="Scan head pairs at selected token">
    <p className="eyebrow">Find a combination worth exploring</p><h3 ref={heading} tabIndex={-1}>Which head pairs increase or suppress this token?</h3>
    <p>Measure each pair and both heads alone at story step {props.source.step}, using the same starting text and captured settings. Rank the interaction gap or the combined probability change.</p>
    <div className="continuation-actions"><label>Token to measure<select aria-label="Pair scan target token" value={props.targetTokenId} disabled={props.disabled && !busy} onChange={event => { if (busy) stop(); props.onTargetChange(Number(event.target.value)); }}>{targetOptions}</select></label><label>Pair strengths<select aria-label="Pair scan strengths" value={strengths} disabled={busy || props.disabled} onChange={event => { setStrengths(event.target.value as PairStrengths); setRows([]); setSelectedKey(''); setAll(false); setError(''); setStatus('Ready to scan.'); }}>{pairStrengthOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
      <button type="button" disabled={busy || props.disabled} onClick={() => void scan()}>Scan head pairs</button><button type="button" disabled={!busy} onClick={stop}>Stop pair scan</button>
      <label>Sort pairs<select aria-label="Sort pair scan results" value={order} onChange={event => setOrder(event.target.value as PairScanOrder)}><option value="interaction-absolute">Strongest interaction (absolute)</option><option value="interaction-positive">Most positive interaction</option><option value="interaction-negative">Most negative interaction</option><option value="absolute">Largest absolute change</option><option value="increase">Largest increase (encourage)</option><option value="decrease">Largest decrease (suppress)</option></select></label></div>
    <p className="hint">The measured token may differ from the generated token. Branching uses the model’s actual choice; it does not force the measured token. Changes above zero encourage it; changes below zero suppress it.</p>
    <p className="hint">Interaction = combined change − sum of individual changes. Positive means a greater increase (or smaller decrease) than the sum; negative means a smaller increase (or greater decrease). A strong interaction can accompany a small net change.</p>
    <p className="hint">{total} combinations in this scope. Mixed strengths test both assignments. All strengths test all nine strength assignments per distinct head pair and take longer. Normal may restore a changed head. Individual head/strength trials are reused within this scan.</p>
    <p role="status" data-testid="pair-scan-status">{status} {rows.length} / {total} combinations measured.</p>
    {busy && <progress value={rows.length} max={total} aria-label="Head pair scan progress" />}
    {error && <p role="alert" className="error">Could not finish the pair scan: {error}</p>}
    {!!rows.length && <p>Selected token: <strong>{tokenLabel(props.tokenizer, props.targetTokenId)}</strong> · Source probability {(rows[0].before*100).toFixed(4)}%.</p>}
    {started && <HeadPairHeatmap key={strengths} rows={rows} config={props.manifest.config} strengths={strengths} disabled={busy || props.disabled} selectedKey={selectedKey} onSelect={row => setSelectedKey(headPairKey(row.edits))} onChoose={choose} />}
    {!!rows.length && <><h4>Ranked pairs</h4>
      <ol className="head-scan-results" aria-label="Ranked head pair changes">{visible.map(row => <li key={`${key(row.edits[0])}|${key(row.edits[1])}`} data-pair-first={key(row.edits[0])} data-pair-second={key(row.edits[1])} className={headPairKey(row.edits) === selectedKey ? 'pair-scan-selected' : ''}>
        <div><strong>{name(row.edits[0])} + {name(row.edits[1])}</strong><p>Both together: <span data-testid="pair-scan-after">{(row.after*100).toFixed(4)}%</span> · <b data-testid="pair-scan-delta">{points(row.delta)}</b> · rank {row.rank}</p>
          <p className="pair-scan-interaction">Interaction: <strong data-testid="pair-scan-interaction">{points(row.interaction)}</strong> · Sum of individual changes: <span data-testid="pair-scan-sum">{points(row.sumDelta)}</span></p>
          <p className="hint">First head alone: <span data-testid="pair-scan-first-delta">{points(row.individualProbabilities[0]-row.before)}</span> · Second head alone: <span data-testid="pair-scan-second-delta">{points(row.individualProbabilities[1]-row.before)}</span></p><small>{row.choice === props.source.choice.tokenId ? 'Same next token' : `Different next token: ${tokenLabel(props.tokenizer, row.choice)}`}</small></div>
        <button type="button" disabled={busy || props.disabled} onClick={() => choose(row)} aria-label={`Use pair layer ${row.edits[0].layer+1} head ${row.edits[0].head+1} strength ${row.edits[0].factor} and layer ${row.edits[1].layer+1} head ${row.edits[1].head+1} strength ${row.edits[1].factor}`}>Use this pair</button>
      </li>)}</ol>
      {rows.length > 8 && <div className="continuation-actions"><button type="button" disabled={busy} onClick={() => setAll(!all)}>{all ? 'Show strongest eight pairs' : `Show all ${rows.length} pairs`}</button></div>}
      <p className="hint">Raw model probability changes before sampling, relative to the captured source. Interaction measures non-additivity for this token and input, including the probability read-out. It does not establish a general head role. Rankings cover only the chosen strengths and completed trials; partial results cannot establish the strongest unmeasured pair. Use a pair to preview its individual and combined effects in the branch controls above, choose duration and token limit, then Compare from this token.</p></>}
  </section>;
}
