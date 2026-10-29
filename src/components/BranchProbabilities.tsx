import { useEffect, useState, type ReactNode } from 'react';
import type { BranchProbabilityRow } from '../branch-probabilities';
import { forkEditLabel, proposedEdits, type ForkPlan } from '../story-fork';
import { tokenLabel } from '../engine/tokenizer';
import type { BpeDocument } from '../engine/bpe';
import type { ModelClient } from '../workers/client';
import { predictStory } from '../workers/story-runtime';
import { measureStoryProbabilities, type StoryProbabilities as Report } from '../story-probabilities';
import { compactPercent, probabilityDirection } from '../probability-display';
const percent = (n: number) => `${(n * 100).toFixed(4)}%`;
const change = (n: number) => `${n > 0 ? '+' : ''}${(n * 100).toFixed(4)} percentage points`;
export function BranchProbabilities({ plan, tokenizer, client, label: regionLabel = 'Head change probabilities', targetTokenId = plan.source.choice.tokenId, children }: { plan: ForkPlan; tokenizer: BpeDocument; client?: ModelClient; label?: string; targetTokenId?: number; children?: ReactNode }) {
  const [report, setReport] = useState<{ plan: ForkPlan; client: ModelClient; targetTokenId: number; value: Report } | null>(null), [error, setError] = useState(''), [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!client) return;
    let current = true;
    setError('');
    void (async () => {
      const value = await measureStoryProbabilities(plan, (ids, edits) => predictStory(client, ids, edits), () => !current, targetTokenId);
      if (!current || !value) return;
      setReport({ plan, client, targetTokenId, value });
    })().catch(e => { if (current) setError(String(e)); });
    return () => { current = false; };
  }, [plan, client, retry, targetTokenId]);
  const data = report?.plan === plan && report.client === client && report.targetTokenId === targetTokenId ? report.value : null;
  const label = (id: number) => tokenLabel(tokenizer, id);
  function row(r: BranchProbabilityRow, selected = false) {
    return <li key={r.tokenId} className={selected ? 'probability-selected' : ''} data-token-id={r.tokenId}>
      <div className="branch-probability-token"><strong>{label(r.tokenId)}</strong><span>{selected ? 'Selected token' : 'Alternative'} · rank {r.beforeRank} → {r.afterRank}</span></div>
      <div className="branch-probability-values"><span data-testid="probability-before">{percent(r.before)}</span><span aria-hidden="true">→</span><span data-testid="probability-after">{percent(r.after)}</span><b data-testid="probability-change">{change(r.delta)}</b></div>
      <div className="branch-probability-bars" aria-hidden="true"><i style={{ width: percent(r.before) }} /><i style={{ width: percent(r.after) }} /></div>
    </li>;
  }
  return <section className="branch-probabilities" aria-label={regionLabel}>
    <p className="eyebrow">{regionLabel === 'Scanned head probability preview' ? 'Preview for a new comparison' : 'What changed at the branch point?'}</p>
    {error ? <p role="alert">Could not measure this change: {error} <button type="button" onClick={() => setRetry(n => n + 1)}>Retry probability comparison</button></p> : !data ? <p role="status">Measuring the branch-point probabilities…</p> : <div className="branch-change-summary" data-testid="branch-change-summary">
      <div><h3>{probabilityDirection(data.selected.delta)}: <span>{label(data.selected.tokenId)}</span></h3>
        <p><strong data-testid="summary-probabilities">{compactPercent(data.selected.before)}% → {compactPercent(data.selected.after)}%</strong>
          <span className="summary-probability-delta" data-testid="summary-probability-change">{data.selected.delta > 0 ? '+' : data.selected.delta < 0 ? '−' : ''}{compactPercent(Math.abs(data.selected.delta))} percentage points</span></p></div>
      <p className="branch-choice-summary">{data.beforeChoice === data.afterChoice ? <>Both versions choose <strong>{label(data.afterChoice)}</strong> at this step.</> : <>Next token changes from <strong>{label(data.beforeChoice)}</strong> to <strong>{label(data.afterChoice)}</strong>.</>}
        <small>Step {plan.source.step} · {proposedEdits(plan).length === 2 ? 'Both head changes together' : 'One head change'} · Same starting text</small></p>
    </div>}
    {children}
    {data && !error && <details className="branch-probability-details">
      <summary>Exact values and alternatives</summary>
      <p>Story step {plan.source.step} · {forkEditLabel(plan)}. Every prediction here uses exactly the same starting text. Values run from <strong>Keep source settings</strong> → <strong>With head change</strong>.</p>
      {data.individuals.length === 2 && <div className="head-pair-effects" aria-label="Individual and combined head effects">
        <p><strong>Probability of {label(data.selected.tokenId)}</strong> · Source: {percent(data.selected.before)}</p>
        <ul aria-label="Individual head effects">{data.individuals.map(({ edit, probability, delta }) => <li key={`${edit.layer}:${edit.head}`} data-pair-layer={edit.layer} data-pair-head={edit.head}><strong>Layer {edit.layer+1} · Head {edit.head+1} alone ({edit.factor}×)</strong><p><span data-testid="individual-probability">{percent(probability)}</span> · <b data-testid="individual-delta">{change(delta)}</b></p></li>)}</ul>
        <p><strong>Both together:</strong> <span data-testid="combined-probability">{percent(data.selected.after)}</span> · <b data-testid="combined-delta">{change(data.selected.delta)}</b></p>
        <p>Sum of individual changes: <span data-testid="sum-individual-deltas">{change(data.individuals.reduce((sum, row) => sum + row.delta, 0))}</span>.</p>
        <p data-testid="head-interaction">Combined minus that sum: <strong>{change(data.interaction)}</strong>. A positive value means a greater increase (or smaller decrease) than the sum; a negative value means a smaller increase (or greater decrease). This comparison describes this token and starting text.</p>
      </div>}
      <ul aria-label="Selected token probability">{row(data.selected, true)}</ul>
      <p data-testid="branch-choice-explanation">{plan.source.settings.temperature === 0 ? 'Most-likely-token selection' : 'Seeded sampling with the recorded temperature, top-p and shared draw'} chooses <strong>{label(data.beforeChoice)}</strong> before this change and <strong>{label(data.afterChoice)}</strong> after it. {data.beforeChoice === data.afterChoice ? 'The next token stays the same even though its probability may change.' : 'This change selects a different next token, which becomes input to later predictions.'} {plan.source.settings.temperature === 0 && data.beforeChoice !== data.afterChoice && data.rows.find(r => r.tokenId === data.afterChoice)!.delta < 0 && 'Here, the winning token’s probability also fell, but it overtook the other candidates.'}</p>
      <p className="hint">Strongest alternatives: the top five model predictions from either version, plus any generated choice. Ranks exclude the beginning and padding tokens.</p>
      <ul aria-label="Strongest token alternatives">{data.rows.filter(r => r.tokenId !== data.selected.tokenId).map(r => row(r))}</ul>
      <p className="hint">These are raw model probabilities before temperature, top-p and generation filtering; bars share a 0–100% scale. This measures the new head edit relative to the source’s captured settings. Head effects can interact. Values are rounded to four decimal places. Later predictions may also reflect ongoing head edits and, once tokens diverge, different input text.</p>
    </details>}
  </section>;
}
