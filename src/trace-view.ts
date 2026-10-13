import { logSoftmax } from './engine/comparison';
import type { EffectTraceReport, TraceMeasurement, TraceReadout } from './engine/trace';
import { traceNodeKey } from './engine/trace-topology';
import { nodeKey, stageNames, type FlowNode } from './flow';

export interface TraceStep { node: FlowNode; position: number; title: string }
export interface TraceView { report: EffectTraceReport; position: number; origin: number; measurements: Map<string, TraceMeasurement> }
export const changeLabel = { exact: 'Exact match', small: 'Small difference', changed: 'Changed' };
export const routeLabel = { unaffected: 'Outside the edit route', prediction: 'Can affect this prediction', local: 'Local downstream computation' };
export const traceNumber = (value: number) => value === 0 ? '0' : Math.abs(value) < .001 ? value.toExponential(2) : value.toPrecision(3);
export const measurementFor = (view: TraceView, node: FlowNode, weights = false) => view.measurements.get(traceNodeKey(
  node.kind === 'head' ? { stage: weights ? 'attention.weights' : 'attention.head.modified', layer: node.layer, head: node.head, position: view.position }
    : node.kind === 'stage' ? { layer: node.layer, stage: node.stage, position: view.position }
    : { stage: node.kind === 'embedding' ? 'embedding.output' : 'logits', position: node.kind === 'readout' ? view.report.predictionPosition : view.position }));
export const readoutFor = (view: TraceView, node: FlowNode) => view.report.readouts.find(row => node.kind === 'readout' ? row.prediction
  : node.kind === 'embedding' ? row.stage === 'embedding.output' : node.kind === 'stage' && row.layer === node.layer && row.stage === node.stage);
export function traceSteps(report: EffectTraceReport, origin: number): TraceStep[] {
  const steps: TraceStep[] = [], patch = report.intervention;
  let position = origin;
  const stage = (layer: number, key: 'block.input' | 'attention.normalized' | 'attention.projected' | 'attention.residual' | 'mlp.output' | 'block.output') =>
    steps.push({ node: { kind: 'stage', layer, stage: key }, position, title: `Layer ${layer + 1} · ${stageNames[key]}` });
  for (let layer = patch.layer; layer < report.config.layers; layer++) {
    if (layer === patch.layer) steps.push({ node: { kind: 'head', layer, head: patch.head }, position, title: `Layer ${layer + 1} · Head ${patch.head + 1} output` });
    else {
      stage(layer, 'block.input'); stage(layer, 'attention.normalized');
      if (report.hasComputationalPath && position <= report.predictionPosition) position = report.predictionPosition;
      for (let head = 0; head < report.config.heads; head++) steps.push({ node: { kind: 'head', layer, head }, position, title: `Layer ${layer + 1} · parallel Head ${head + 1} output` });
    }
    stage(layer, 'attention.projected'); stage(layer, 'attention.residual'); stage(layer, 'mlp.output'); stage(layer, 'block.output');
  }
  steps.push({ node: { kind: 'readout' }, position: report.predictionPosition, title: 'Final readout · model prediction' });
  return steps;
}
export const matchingStep = (steps: TraceStep[], node: FlowNode, position: number) => steps.findIndex(step => nodeKey(step.node) === nodeKey(node) && step.position === position);
export function readoutProbabilities(logits: number[], target: number) {
  const logProbabilities = logSoftmax(logits), probabilities = Array.from(logProbabilities, Math.exp);
  const top = probabilities.reduce((best, value, token) => value > probabilities[best] ? token : best, 0);
  return { top, topProbability: probabilities[top], targetProbability: probabilities[target], targetLogProbability: logProbabilities[target] };
}
export const firstReadoutFlip = (report: EffectTraceReport) => report.readouts.find(row => readoutProbabilities(row.baselineLogits, 0).top !== readoutProbabilities(row.modifiedLogits, 0).top);
export function handoff(report: EffectTraceReport, origin?: number) {
  const patch = report.intervention;
  const from = patch.tokens.kind === 'all' ? origin : patch.tokens.indices.length === 1 ? patch.tokens.indices[0] : undefined;
  if (!report.hasComputationalPath || from === undefined || from >= report.predictionPosition || patch.layer >= report.config.layers - 1) return null;
  return { layer: patch.layer + 1, from, to: report.predictionPosition, allPositions: patch.tokens.kind === 'all' };
}
export function readoutTitle(row: TraceReadout) {
  return row.prediction ? 'Model prediction' : row.stage === 'embedding.output' ? 'Input embedding' : `Layer ${row.layer! + 1} ${row.stage === 'attention.residual' ? 'after attention' : 'block output'}`;
}
