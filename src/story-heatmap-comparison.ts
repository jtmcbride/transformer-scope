import { selectStoryToken, type StorySelection } from './continuation';
import { selectForkToken, type ForkPredict } from './story-fork';
import type { ModelConfig } from './engine/contracts';
import type { StoryWorkspace } from './story-workspace';
import { headPairKey } from './head-pair-heatmap';
import { scanStoryHeadPairs, type HeadPairScanRow, type PairStrengths } from './story-head-pair-scan';

export function workspaceStoryPaths(workspace: StoryWorkspace) {
  const paths: { id: string; label: string; nodeId: string | null; count: number; start: number; select: (index: number) => StorySelection }[] = [];
  const root = workspace.root;
  if (!root) return paths;
  for (const branch of ['original','edited'] as const) if (root.frame[branch]?.choices.length) paths.push({
    id: JSON.stringify([null,branch]), label: `Source story · ${branch === 'original' ? 'Original' : 'Edited'}`, nodeId: null,
    count: root.frame[branch]!.choices.length, start: 1,
    select: index => selectStoryToken(root.frame,root.tokens.length,branch,index,root.settings,root.edit),
  });
  workspace.nodes.forEach((node,i) => {
    const groupIndex = workspace.sharedForks?.findIndex(group => group.nodeIds.includes(node.id)) ?? -1;
    const group = workspace.sharedForks?.[groupIndex];
    const sharedLabel = group ? ` · Shared ${groupIndex+1} ${group.nodeIds[0] === node.id ? 'A' : 'B'}` : '';
    for (const branch of ['original','edited'] as const) if (node.frame[branch]?.choices.length) paths.push({
      id: JSON.stringify([node.id,branch]), label: `Comparison ${i+1}${sharedLabel} · ${branch === 'original' ? 'Keep source settings' : 'With head change'}`, nodeId: node.id,
      count: node.frame[branch]!.choices.length, start: node.plan.source.step,
      select: index => selectForkToken(node.plan,node.frame,branch,index),
    });
  });
  return paths;
}
export interface InteractionChange { key: string; a: HeadPairScanRow; b: HeadPairScanRow; difference: number }
/** Only compare completed matching head/strength assignments. Missing never means zero. */
export function compareInteractions(a: HeadPairScanRow[], b: HeadPairScanRow[]): InteractionChange[] {
  const byKey = new Map(b.map(row => [headPairKey(row.edits),row]));
  return a.flatMap(row => { const key = headPairKey(row.edits), other = byKey.get(key); return other ? [{ key, a: row, b: other, difference: other.interaction-row.interaction }] : []; });
}
export function rankInteractionChanges(rows: InteractionChange[], order: 'absolute' | 'increase' | 'decrease') {
  const score = (row: InteractionChange) => order === 'absolute' ? Math.abs(row.difference) : order === 'increase' ? row.difference : -row.difference;
  return [...rows].sort((a,b) => score(b)-score(a) || a.a.edits[0].layer-b.a.edits[0].layer || a.a.edits[0].head-b.a.edits[0].head || a.a.edits[1].layer-b.a.edits[1].layer || a.a.edits[1].head-b.a.edits[1].head || a.a.edits[0].factor-b.a.edits[0].factor || a.a.edits[1].factor-b.a.edits[1].factor);
}
export type WithSource = (source: StorySelection, use: (predict: ForkPredict) => Promise<void>) => Promise<void>;
/** Each source retains its own prefix, replay check, inherited edits, and global sampling counter. */
export async function scanStoryHeatmaps(sources: [StorySelection,StorySelection], config: ModelConfig, strengths: PairStrengths, withSource: WithSource,
  emit: (side: 0 | 1, row: HeadPairScanRow) => void, cancelled: () => boolean) {
  const captured = structuredClone(sources);
  for (const side of [0,1] as const) {
    if (cancelled()) return;
    await withSource(captured[side],predict => scanStoryHeadPairs(captured[side],config,strengths,predict,row => { if (!cancelled()) emit(side,row); },cancelled));
  }
}
