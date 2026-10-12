import type { HeadScaleIntervention } from './contracts';

export type ExecutionPreference = 'auto' | 'full';
export type ExecutionPlan =
  | { strategy: 'identity'; fromLayer: null }
  | { strategy: 'full'; fromLayer: 0 }
  | { strategy: 'cached-suffix'; fromLayer: number; patches: HeadScaleIntervention[] };

export function effectiveInterventions(patches: readonly HeadScaleIntervention[]): HeadScaleIntervention[] {
  return patches.filter(p => p.factor !== 1 && (p.tokens.kind === 'all' || p.tokens.indices.length > 0));
}

/** Requests are validated before reaching this pure planner. */
export function planExecution(patches: readonly HeadScaleIntervention[], cacheAvailable: boolean, preference: ExecutionPreference = 'auto'): ExecutionPlan {
  if (preference === 'full') return { strategy: 'full', fromLayer: 0 };
  const effective = effectiveInterventions(patches);
  if (!effective.length) return { strategy: 'identity', fromLayer: null };
  const layer = effective[0].layer;
  if (cacheAvailable && effective.every(p => p.layer === layer)) return { strategy: 'cached-suffix', fromLayer: layer, patches: effective };
  return { strategy: 'full', fromLayer: 0 };
}
