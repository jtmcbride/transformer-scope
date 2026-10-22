import { useEffect, useState } from 'react';
import { branchProbabilities, type BranchProbabilities as Report, type BranchProbabilityRow } from '../branch-probabilities';
import { forkEdits, type ForkPlan } from '../story-fork';
import { tokenLabel } from '../engine/tokenizer';
import type { BpeDocument } from '../engine/bpe';
import type { ModelClient } from '../workers/client';
import { predictStory } from '../workers/story-runtime';
const percent = (n: number) => `${(n * 100).toFixed(4)}%`;
const change = (n: number) => `${n > 0 ? '+' : ''}${(n * 100).toFixed(4)} percentage points`;
export function BranchProbabilities({ plan, tokenizer, client, label: regionLabel = 'Head change probabilities' }: { plan: ForkPlan; tokenizer: BpeDocument; client?: ModelClient; label?: string }) {
  const [report, setReport] = useState<{ plan: ForkPlan; client: ModelClient; value: Report } | null>(null), [error, setError] = useState(''), [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!client) return;
    let current = true;
    setError('');
    void (async () => {
      const before = await predictStory(client, plan.source.prefix, forkEdits(plan, false, 0));
      if (!current) return;
      const after = await predictStory(client, plan.source.prefix, forkEdits(plan, true, 0));
      if (!current) return;
      const value = branchProbabilities(before, after, plan.source.choice.tokenId, plan.source.settings, plan.source.step);
      if (value.beforeChoice !== plan.source.choice.tokenId || Math.abs(value.selected.before - plan.source.choice.probability) > 1e-6) throw new Error('The source prediction could not be reproduced.');
      setReport({ plan, client, value });
    })().catch(e => { if (current) setError(String(e)); });
    return () => { current = false; };
  }, [plan, client, retry]);
  const data = report?.plan === plan && report.client === client ? report.value : null;
  const label = (id: number) => tokenLabel(tokenizer, id);
  function row(r: BranchProbabilityRow, selected = false) {
    return <li key={r.tokenId} className={selected ? 'probability-selected' : ''} data-token-id={r.tokenId}>
      <div className="branch-probability-token"><strong>{label(r.tokenId)}</strong><span>{selected ? 'Selected token' : 'Alternative'} · rank {r.beforeRank} → {r.afterRank}</span></div>
      <div className="branch-probability-values"><span data-testid="probability-before">{percent(r.before)}</span><span aria-hidden="true">→</span><span data-testid="probability-after">{percent(r.after)}</span><b data-testid="probability-change">{change(r.delta)}</b></div>
      <div className="branch-probability-bars" aria-hidden="true"><i style={{ width: percent(r.before) }} /><i style={{ width: percent(r.after) }} /></div>
    </li>;
  }
  return <section className="branch-probabilities" aria-label={regionLabel}>
    <p className="eyebrow">{regionLabel === 'Scanned head probability preview' ? 'Preview for a new comparison' : 'At the point where the story can branch'}</p><h3>How this head change shifts the prediction</h3>
    <p>Story step {plan.source.step} · Layer {plan.edit.layer + 1}, Head {plan.edit.head + 1} · {plan.edit.factor.toFixed(2)}×. Both predictions use exactly the same starting text. Values run from <strong>Keep source settings</strong> → <strong>With head change</strong>.</p>
    {error ? <p role="alert">Could not measure this change: {error} <button type="button" onClick={() => setRetry(n => n + 1)}>Retry probability comparison</button></p> : !data ? <p role="status">Measuring the branch-point probabilities…</p> : <>
      <ul aria-label="Selected token probability">{row(data.selected, true)}</ul>
      <p data-testid="branch-choice-explanation">{plan.source.settings.temperature === 0 ? 'Most-likely-token selection' : 'Seeded sampling with the recorded temperature, top-p and shared draw'} chooses <strong>{label(data.beforeChoice)}</strong> before this change and <strong>{label(data.afterChoice)}</strong> after it. {data.beforeChoice === data.afterChoice ? 'The next token stays the same even though its probability may change.' : 'This change selects a different next token, which becomes input to later predictions.'} {plan.source.settings.temperature === 0 && data.beforeChoice !== data.afterChoice && data.rows.find(r => r.tokenId === data.afterChoice)!.delta < 0 && 'Here, the winning token’s probability also fell, but it overtook the other candidates.'}</p>
      <p className="hint">Strongest alternatives: the top five model predictions from either version, plus any generated choice. Ranks exclude the beginning and padding tokens.</p>
      <ul aria-label="Strongest token alternatives">{data.rows.filter(r => r.tokenId !== data.selected.tokenId).map(r => row(r))}</ul>
      <p className="hint">These are raw model probabilities before temperature, top-p and generation filtering; bars share a 0–100% scale. This measures the new head edit relative to the source’s captured settings. Head effects can interact. Values are rounded to four decimal places. Later predictions may also reflect ongoing head edits and, once tokens diverge, different input text.</p>
    </>}
  </section>;
}
