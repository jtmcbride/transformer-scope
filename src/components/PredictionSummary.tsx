import type { ComparisonResult } from '../engine/runs';

interface Props { comparison: ComparisonResult; target: number; symbol: string; position: number; pending: boolean; ended: boolean; disabled: boolean; onReset: () => void }
export function PredictionSummary(props: Props) {
  const baseline = props.comparison.baselineTopTokens[0], modified = props.comparison.modifiedTopTokens[0];
  const target = props.comparison.displayTokens.find(row => row.tokenId === props.target)!;
  return <div className={`prediction-summary ${props.pending ? 'result-pending' : ''}`} aria-label="Prediction summary" aria-busy={props.pending}>
    <div className="summary-heading"><span className="eyebrow">{props.ended ? 'Last completed result · session ended' : props.pending ? 'Last completed result · updating' : `Next token · after position ${props.position}`}</span>
      <button type="button" className="text-button" disabled={props.disabled} onClick={props.onReset}>Reset to baseline</button></div>
    <div className="answer-comparison"><div><span>Baseline</span><strong data-testid="baseline-answer">{baseline.text}</strong></div><span className="answer-arrow" aria-hidden="true">→</span>
      <div><span>Modified</span><strong data-testid="modified-answer">{modified.text}</strong></div>
      <div className="target-comparison"><span>Probability of {props.symbol}</span><strong data-testid="target-probability">{(target.baselineProbability * 100).toFixed(2)}% <span aria-hidden="true">→</span> {(target.probability * 100).toFixed(2)}%</strong></div></div>
  </div>;
}
