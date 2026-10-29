import type { BpeDocument } from './engine/bpe';
import { tokenLabel } from './engine/tokenizer';
import { proposedEdits } from './story-fork';
import type { StoryNode, StoryWorkspace } from './story-workspace';

export function workspaceName(data: StoryWorkspace): string {
  if (data.name) return data.name;
  const text = data.root?.prompt.trim().replace(/\s+/gu, ' ');
  return text ? text.length > 64 ? `${text.slice(0, 61)}…` : text : 'New story';
}
export function branchName(node: StoryNode, tokenizer: BpeDocument): string {
  if (node.name) return node.name;
  const edits = proposedEdits(node.plan), factor = edits[0].factor;
  const action = edits.every(edit => edit.factor === factor) ? factor === 0 ? 'Muted' : factor === 1 ? 'Restored' : factor === 2 ? 'Amplified' : 'Scaled' : 'Changed';
  const heads = edits.map(edit => `L${edit.layer + 1}/H${edit.head + 1}${action === 'Scaled' || action === 'Changed' ? ` ${edit.factor}×` : ''}`).join(' + ');
  const token = tokenLabel(tokenizer, node.plan.source.choice.tokenId).replaceAll('␠', ' ').trim() || '(space)';
  return `${action} ${heads} at “${token}”`;
}
