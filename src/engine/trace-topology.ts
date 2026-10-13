import type { ActivationReference, AttentionOverrideIntervention, HeadScaleIntervention, ModelConfig } from './contracts';

export type TraceIntervention = HeadScaleIntervention | AttentionOverrideIntervention;
export const tracePositions = (patch: TraceIntervention, length: number) => patch.kind === 'attention.override' ? [patch.query]
  : patch.tokens.kind === 'all' ? Array.from({ length }, (_, i) => i) : patch.tokens.indices;

export interface TraceAddress extends ActivationReference { position: number; head?: number }
export type TraceRoute = 'unaffected' | 'prediction' | 'local';
export const traceNodeKey = (node: TraceAddress) => `${node.layer ?? '-'}:${node.stage}:${node.position}:${node.head ?? '-'}`;

/** Architecture-only reachability. Callers validate the intervention, position, and config first.
 * Attention mixes earlier sources into queries; every other operation preserves position.
 * Head scaling starts after raw outputs; a redirect starts at one post-softmax attention row.
 * Reading a source value does not edit the computation at that source position.
 */
export function traceRoute(node: TraceAddress, patch: TraceIntervention, config: ModelConfig, length: number, prediction: number): TraceRoute {
  if (patch.kind === 'head.scale' && patch.factor === 1) return 'unaffected';
  const positions = tracePositions(patch, length);
  const first = Math.min(...positions), atRoot = positions.includes(node.position);
  let downstream = false;
  if (node.stage === 'embedding.output' || node.layer !== undefined && node.layer < patch.layer) return 'unaffected';
  if (node.layer === undefined) {
    downstream = atRoot || (node.position >= first && patch.layer < config.layers - 1);
  } else if (node.layer === patch.layer) {
    if (node.stage === 'attention.head.modified') downstream = node.head === patch.head && atRoot;
    else if (patch.kind === 'attention.override' && ['attention.weights', 'attention.head.raw'].includes(node.stage)) downstream = node.head === patch.head && atRoot;
    else if (['attention.projected', 'attention.residual', 'mlp.output', 'block.output'].includes(node.stage)) downstream = atRoot;
  } else {
    // The next layer's inputs are still local; its attention is the first opportunity to mix positions.
    downstream = ['block.input', 'attention.normalized'].includes(node.stage)
      ? atRoot || (node.layer > patch.layer + 1 && node.position >= first)
      : node.position >= first;
  }
  if (!downstream) return 'unaffected';
  const canContinue = node.layer === undefined ? node.position === prediction
    : ['block.input', 'attention.normalized'].includes(node.stage) || node.layer < config.layers - 1
      ? node.position <= prediction : node.position === prediction;
  return canContinue ? 'prediction' : 'local';
}
