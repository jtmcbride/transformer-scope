import type { ExperimentRecipe } from '../experiments';
import type { ComparisonResult } from '../engine/runs';
import { traceNumber } from '../trace-view';

const questions: Record<string, string> = {
  effect: 'What changes when this head is muted?',
  small: 'Can a route exist with very little effect?',
  unreachable: 'Where does a local change stop?',
};
const prompts: Record<string, string> = {
  effect: 'Follow the residual readouts to find where the modified preferred token first changes.',
  small: 'Compare probabilities and Δ log p, then follow the position handoff. A route alone does not determine effect size.',
  unreachable: 'Inspect the edited position, then the prediction position. A local vector can change while the prediction stays the same.',
};
export function recipeEdit(recipe: ExperimentRecipe) {
  const patch = recipe.intervention;
  return `Layer ${patch.layer + 1} · Head ${patch.head + 1} · ${patch.factor}× · ${patch.tokens.kind === 'all' ? 'all positions' : `position ${patch.tokens.indices.join(', ')}`} · predict after ${recipe.predictionPosition}`;
}

export function GuidedExperiments({ recipes, completed, launching, disabled, draft, onLaunch }: {
  recipes: ExperimentRecipe[]; completed?: string; launching?: string | null; disabled: boolean; draft: boolean; onLaunch: (id: string) => void;
}) {
  return <section className="guided-experiments" aria-label="Prepared examples">
    <div className="examples-heading"><h3>Try an example</h3><p>Choose a question to explore, then inspect the edit and follow its result.</p></div>
    <div className="example-cards">{recipes.map((recipe, index) => <button type="button" key={recipe.id}
      className={`example-card ${completed === recipe.id ? 'example-completed' : ''}`} disabled={disabled}
      aria-label={`Start example: ${recipe.title}`} aria-pressed={completed === recipe.id} onClick={() => onLaunch(recipe.id)}>
      <span className="example-order">{String(index + 1).padStart(2, '0')}<span>{launching === recipe.id ? 'Loading example…' : completed === recipe.id ? 'Completed example' : 'Start example →'}</span></span>
      <strong>{recipe.title}</strong><span>{questions[recipe.id] ?? 'Follow this edit through the computation.'}</span>
      <small>{recipeEdit(recipe)}</small>
    </button>)}</div>
    <p className="examples-note">These examples were selected to illustrate different outcomes. Starting an example recomputes it with the installed model{draft ? ' and replaces your input or source draft' : ''}.</p>
  </section>;
}

export function ExperimentGuide({ recipe, comparison, target, disabled, pending, ended, draft, traceOpen, onInspect, onTrace }: {
  recipe?: ExperimentRecipe; comparison: ComparisonResult; target: number; disabled: boolean; pending: boolean; ended: boolean; draft: boolean; traceOpen: boolean;
  onInspect: () => void; onTrace: () => void;
}) {
  if (!recipe) return <div className="experiment-guide custom-guide" data-testid="example-guide">
    <p className="eyebrow">Custom experiment{pending ? ' · showing the last completed result' : ended ? ' · session ended' : ''}</p>
    <p>Inspect a node, change a head, redirect attention, or copy a residual vector. Follow the change measures the completed edit.</p>
  </div>;
  const probability = comparison.displayTokens.find(token => token.tokenId === target);
  return <section className="experiment-guide" aria-label="Example guide" data-testid="example-guide">
    <p className="eyebrow">{ended ? 'Last completed example · session ended' : pending ? 'Last completed example · waiting for computation' : 'Completed example'}</p>
    <h3>{recipe.title}</h3><p className="example-edit">{recipeEdit(recipe)}</p>
    {draft && <p className="example-draft" role="status">Draft choices have not been applied. This guide describes the completed example.</p>}
    <p className="example-result" data-testid="example-result">Measured preferred token: <strong>{comparison.baselineTopTokens[0].text} → {comparison.modifiedTopTokens[0].text}</strong>
      {probability && <> · p({probability.text}): {(probability.baselineProbability * 100).toFixed(4)}% → {(probability.probability * 100).toFixed(4)}%</>}
      {comparison.metrics.targetLogProbabilityDelta !== undefined && <> · Δ log p: <span>{traceNumber(comparison.metrics.targetLogProbabilityDelta)} nats</span></>}
    </p>
    <p className="example-route">{comparison.hasComputationalPath ? 'The edit has a possible computational route to this prediction.' : 'The edited position has no computational route to this prediction.'}</p>
    <p>{prompts[recipe.id] ?? recipe.interpretation}</p>
    <div className="example-guide-actions">
      <button type="button" className="secondary" disabled={disabled} onClick={onInspect}>Inspect the edited head</button>
      <button type="button" disabled={disabled} onClick={onTrace}>{traceOpen ? 'Return to trace start' : 'Trace this example'} <span aria-hidden="true">→</span></button>
    </div>
  </section>;
}
