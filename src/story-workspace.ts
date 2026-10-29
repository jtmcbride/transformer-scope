import { StoryContinuation, selectStoryToken, type StoryFrame, type Sampling } from './continuation';
import { StoryFork, selectForkToken, type ForkPlan } from './story-fork';
import { requireInput, validateTokens, validateInterventions, type HeadScaleIntervention, type ModelManifest } from './engine/contracts';
import { SharedStoryFork } from './story-shared-fork';
export interface StoryRoot { prompt: string; tokens: number[]; settings: Sampling; edit?: HeadScaleIntervention; frame: StoryFrame }
export interface StoryNode { id: string; parentId: string | null; plan: ForkPlan; frame: StoryFrame; name?: string }
export interface SharedStoryGroup { id: string; nodeIds: [string, string]; labels: [string, string]; targetTokenId?: number }
export interface StoryWorkspace { format: 'transformer-story-workspace'; version: 1 | 2 | 3; checkpoint: string; root: StoryRoot | null; nodes: StoryNode[]; route: string[]; sharedForks?: SharedStoryGroup[]; name?: string }
export const workspaceNameLimit = 80;
/** Streaming computation updates must retain the user's current name, including clearing it. */
export function mergeStoryNode(current: StoryNode, incoming: StoryNode): StoryNode {
  const { name: _capturedName, ...record } = incoming;
  return { ...current, ...record };
}
/** Presentation names never invalidate a measured distribution or captured branch setup. */
export function storyAnalysisKey(value: unknown): string { return JSON.stringify(value, (key, item) => key === 'name' || key === 'labels' ? undefined : item) ?? 'undefined'; }
function validateName(name: unknown) { requireInput(name === undefined || typeof name === 'string' && name.trim().length > 0 && name.length <= workspaceNameLimit, 'Names must contain 1–80 characters.'); }
export function checkpointKey(m: ModelManifest) { return [m.modelId, m.modelRevision, m.weightsSha256, m.tokenizerSha256].join(':'); }
export function emptyWorkspace(m: ModelManifest): StoryWorkspace { return { format: 'transformer-story-workspace', version: 3, checkpoint: checkpointKey(m), root: null, nodes: [], route: [], sharedForks: [] }; }
export function nodeRoute(nodes: StoryNode[], id: string): string[] {
  const route: string[] = [];
  let node = nodes.find(n => n.id === id);
  while (node) { requireInput(!route.includes(node.id), 'Comparison links contain a cycle.'); route.unshift(node.id); node = nodes.find(n => n.id === node!.parentId); }
  return route;
}
/** Validate before replacing any saved state. Restored computation is separately replay-checked by the inspector. */
export function parseWorkspace(value: unknown, manifest: ModelManifest): StoryWorkspace {
  const w = value as StoryWorkspace;
  requireInput(!!w && w.format === 'transformer-story-workspace' && [1,2,3].includes(w.version), 'Unsupported story workspace file.');
  requireInput(w.checkpoint === checkpointKey(manifest), 'This workspace uses a different checkpoint.');
  validateName(w.name);
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
    validateName(node.name);
    requireInput(w.version >= 2 || node.plan.additionalEdits === undefined, 'Two-head comparisons need workspace version 2 or later.');
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
  const groups = w.sharedForks === undefined ? [] : w.sharedForks;
  requireInput(Array.isArray(groups) && groups.length <= 250 && (w.version === 3 || groups.length === 0), 'Shared story comparisons need workspace version 3.');
  const groupIds = new Set<string>(), members = new Set<string>();
  for (const group of groups) {
    requireInput(!!group && typeof group.id === 'string' && group.id.length > 0 && group.id.length <= 100 && !groupIds.has(group.id), 'Invalid shared comparison ID.');
    requireInput(Array.isArray(group.nodeIds) && group.nodeIds.length === 2 && group.nodeIds.every(id => typeof id === 'string' && seen.has(id) && !members.has(id)) && group.nodeIds[0] !== group.nodeIds[1], 'Invalid shared comparison links.');
    requireInput(Array.isArray(group.labels) && group.labels.length === 2 && group.labels.every(label => typeof label === 'string' && label.length <= 200), 'Invalid shared comparison labels.');
    requireInput(group.targetTokenId === undefined || Number.isInteger(group.targetTokenId) && group.targetTokenId >= 1 && group.targetTokenId !== 2 && group.targetTokenId < manifest.config.vocabularySize, 'Invalid shared comparison target token.');
    const nodes = group.nodeIds.map(id => w.nodes.find(node => node.id === id)!);
    const shared = new SharedStoryFork([nodes[0].plan,nodes[1].plan], manifest.config);
    shared.restore([nodes[0].frame,nodes[1].frame]);
    groupIds.add(group.id); group.nodeIds.forEach(id => members.add(id));
  }
  requireInput(w.route.every(id => typeof id === 'string') && (w.route.length === 0 || JSON.stringify(nodeRoute(w.nodes, w.route.at(-1)!)) === JSON.stringify(w.route)), 'Invalid selected comparison path.');
  return structuredClone({ ...w, version: 3, sharedForks: groups });
}
