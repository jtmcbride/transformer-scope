import { useEffect, useMemo, useRef, useState } from 'react';
import type { BpeDocument } from '../engine/bpe';
import { forkEditLabel } from '../story-fork';
import { workspaceNameLimit, type StoryNode, type StoryWorkspace } from '../story-workspace';
import { branchName } from '../story-workspace-names';

export function EditStoryName({ value, suggested, label, onSave, disabled = false }: { value?: string; suggested: string; label: string; onSave: (value?: string) => void; disabled?: boolean }) {
  const [editing, setEditing] = useState(false), [draft, setDraft] = useState(value ?? '');
  const trigger = useRef<HTMLButtonElement>(null), returnFocus = useRef(false);
  useEffect(() => { if (!editing && returnFocus.current) { trigger.current?.focus(); returnFocus.current = false; } }, [editing]);
  function close() { returnFocus.current = true; setEditing(false); }
  if (!editing) return <button ref={trigger} type="button" className="history-rename" disabled={disabled} onClick={() => { setDraft(value ?? ''); setEditing(true); }}>Rename {label}</button>;
  return <form className="history-name-editor" onSubmit={event => { event.preventDefault(); onSave(draft.trim() || undefined); close(); }}>
    <label>{label === 'workspace' ? 'Workspace name' : 'Comparison name'}<input autoFocus aria-label={label === 'workspace' ? 'Workspace name' : 'Comparison name'} value={draft} placeholder={suggested} maxLength={workspaceNameLimit} onChange={event => setDraft(event.target.value)} /></label>
    <button type="submit">Save name</button><button type="button" onClick={close}>Cancel rename</button>
    <small>Up to {workspaceNameLimit} characters. Leave blank to use the suggested name.</small>
  </form>;
}

export function StoryHistory({ data, tokenizer, onNavigate, onRename }: { data: StoryWorkspace; tokenizer: BpeDocument; onNavigate: (id: string) => void; onRename: (id: string, name?: string) => void }) {
  const children = useMemo(() => {
    const map = new Map<string | null, { node: StoryNode; index: number }[]>();
    data.nodes.forEach((node, index) => { const list = map.get(node.parentId) ?? []; list.push({ node, index }); map.set(node.parentId, list); });
    return map;
  }, [data.nodes]);
  const ancestors = new Set(data.route), selected = data.route.at(-1);
  function branch(parent: string | null, depth: number) {
    const rows = children.get(parent);
    return rows?.length ? <ul className={depth > 5 ? 'history-deep' : undefined}>{rows.map(({ node, index }) => {
      const name = branchName(node, tokenizer), current = selected === node.id;
      const shared = data.sharedForks?.findIndex(group => group.nodeIds.includes(node.id)) ?? -1;
      const group = data.sharedForks?.[shared];
      return <li key={node.id} data-history-id={node.id} data-history-parent={node.parentId ?? 'root'} className={ancestors.has(node.id) ? 'history-ancestor' : undefined}>
        <div className="history-row"><button type="button" className="history-open" aria-label={`Comparison ${index + 1}: ${name}`} aria-current={current ? 'location' : undefined} onClick={() => onNavigate(node.id)}>
          <strong>{name}</strong><span>Step {node.plan.source.step} · {node.plan.source.branch === 'original' ? 'from source settings' : 'from the changed path'} · {node.plan.duration === 'prediction' ? 'one prediction' : 'from here onward'}</span>
          <small>Comparison {index + 1} · depth {depth + 1}{group && ` · Shared ${shared + 1} ${group.nodeIds[0] === node.id ? 'A' : 'B'}`} · {forkEditLabel(node.plan)}</small>
          {current && <b className="history-current">Current comparison</b>}
        </button></div>
        {current && <EditStoryName key={node.id} value={node.name} suggested={name} label="comparison" onSave={name => onRename(node.id, name)} />}
        {branch(node.id, depth + 1)}
      </li>;
    })}</ul> : null;
  }
  return <nav className="story-history" aria-label="Story branch map"><ul><li className="history-root">
    <button type="button" className="history-open" aria-label="Source story" aria-current={!selected ? 'location' : undefined} onClick={() => onNavigate('')}><strong>Source story</strong><span>The original prompt and generated paths</span>{!selected && <b className="history-current">Current story</b>}</button>
    {branch(null, 0)}
  </li></ul></nav>;
}
