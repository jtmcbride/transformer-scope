import type { HeadScaleIntervention, Intervention } from './contracts';

export type ExecutionPreference = 'auto' | 'full';
export type ExecutionPlan =
  | { strategy: 'identity'; fromLayer: null }
  | { strategy: 'full'; fromLayer: 0 }
  | { strategy: 'cached-suffix'; fromLayer: number; patches: HeadScaleIntervention[] };

/** Factor-1 and empty head scalings are identities. A residual replacement is always effective: its values may differ from the run's own. */
export function effectiveInterventions(patches: readonly Intervention[]): Intervention[] {
  return patches.filter(p => p.kind === 'residual.replace' || (p.factor !== 1 && (p.tokens.kind === 'all' || p.tokens.indices.length > 0)));
}

/** Requests are validated before reaching this pure planner. */
export function planExecution(patches: readonly Intervention[], cacheAvailable: boolean, preference: ExecutionPreference = 'auto'): ExecutionPlan {
  if (preference === 'full') return { strategy: 'full', fromLayer: 0 };
  const effective = effectiveInterventions(patches);
  if (!effective.length) return { strategy: 'identity', fromLayer: null };
  // The cached suffix covers same-layer head scaling only; residual replacement uses the full forward pass.
  const heads = effective.filter((p): p is HeadScaleIntervention => p.kind === 'head.scale');
  const layer = effective[0].layer;
  if (cacheAvailable && heads.length === effective.length && heads.every(p => p.layer === layer)) return { strategy: 'cached-suffix', fromLayer: layer, patches: heads };
  return { strategy: 'full', fromLayer: 0 };
}
