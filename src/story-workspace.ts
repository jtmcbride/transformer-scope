import { StoryContinuation, selectStoryToken, type StoryFrame, type Sampling } from './continuation';
import { StoryFork, selectForkToken, type ForkPlan } from './story-fork';
import { requireInput, validateTokens, validateInterventions, type HeadScaleIntervention, type ModelManifest } from './engine/contracts';
export interface StoryRoot { prompt: string; tokens: number[]; settings: Sampling; edit?: HeadScaleIntervention; frame: StoryFrame }
export interface StoryNode { id: string; parentId: string | null; plan: ForkPlan; frame: StoryFrame }
export interface StoryWorkspace { format: 'transformer-story-workspace'; version: 1; checkpoint: string; root: StoryRoot | null; nodes: StoryNode[]; route: string[] }
export function checkpointKey(m: ModelManifest) { return [m.modelId, m.modelRevision, m.weightsSha256, m.tokenizerSha256].join(':'); }
export function emptyWorkspace(m: ModelManifest): StoryWorkspace { return { format: 'transformer-story-workspace', version: 1, checkpoint: checkpointKey(m), root: null, nodes: [], route: [] }; }
export function nodeRoute(nodes: StoryNode[], id: string): string[] {
  const route: string[] = [];
  let node = nodes.find(n => n.id === id);
  while (node) { requireInput(!route.includes(node.id), 'Comparison links contain a cycle.'); route.unshift(node.id); node = nodes.find(n => n.id === node!.parentId); }
  return route;
}
/** Validate before replacing any saved state. Restored computation is separately replay-checked by the inspector. */
export function parseWorkspace(value: unknown, manifest: ModelManifest): StoryWorkspace {
  const w = value as StoryWorkspace;
  requireInput(!!w && w.format === 'transformer-story-workspace' && w.version === 1, 'Unsupported story workspace file.');
  requireInput(w.checkpoint === checkpointKey(manifest), 'This workspace uses a different checkpoint.');
  requireInput(Array.isArray(w.nodes) && w.nodes.length <= 500 && Array.isArray(w.route), 'Invalid or oversized workspace.');
  if (w.root) {
    requireInput(typeof w.root.prompt === 'string' && w.root.prompt.length <= 100000, 'Invalid saved prompt.');
    validateTokens(w.root.tokens, manifest.config);
    validateInterventions(w.root.edit ? [w.root.edit] : [], manifest.config, w.root.tokens.length);
    new StoryContinuation(w.root.tokens, w.root.settings, manifest.config.contextLength, w.root.edit).restore(w.root.frame);
    for (const branch of [w.root.frame.original, ...(w.root.frame.edited ? [w.root.frame.edited] : [])]) validateTokens(branch.tokenIds, manifest.config);
  } else requireInput(w.nodes.length === 0, 'Comparisons need a source story.');
  const seen = new Set<string>();
  for (const node of w.nodes) {
    requireInput(typeof node.id === 'string' && node.id.length <= 100 && !seen.has(node.id), 'Invalid or duplicate comparison ID.');
    requireInput(node.parentId === null || seen.has(node.parentId), 'A comparison must follow its parent.');
    const parent = w.nodes.find(n => n.id === node.parentId);
    const source = node.plan.source;
    const expected = parent ? selectForkToken(parent.plan, parent.frame, source.branch, source.step - parent.plan.source.step)
      : selectStoryToken(w.root!.frame, w.root!.tokens.length, source.branch, source.step - 1, w.root!.settings, w.root!.edit);
    requireInput(JSON.stringify(source) === JSON.stringify(expected), 'A comparison source does not match its parent.');
    new StoryFork(node.plan, manifest.config).restore(node.frame);
    for (const branch of [node.frame.original, node.frame.edited!]) validateTokens(branch.tokenIds, manifest.config);
    seen.add(node.id);
  }
  requireInput(w.route.every(id => typeof id === 'string') && (w.route.length === 0 || JSON.stringify(nodeRoute(w.nodes, w.route.at(-1)!)) === JSON.stringify(w.route)), 'Invalid selected comparison path.');
  return structuredClone(w);
}
