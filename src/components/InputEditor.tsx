import { useState, type FormEvent } from 'react';
import type { ModelChoice } from '../experiments';

interface Binding { key: string; value: string }
function retrievalInput(text: string): { bindings: Binding[]; query: string } | null {
  const tokens = text.trim().split(/\s+/u);
  if (tokens.length < 7 || (tokens.length - 3) % 4 !== 0 || tokens.at(-3) !== '?' || tokens.at(-1) !== '=') return null;
  const bindings: Binding[] = [];
  for (let i = 0; i < tokens.length - 3; i += 4) {
    if (!/^[A-D]$/u.test(tokens[i]) || tokens[i + 1] !== '=' || !/^[0-3]$/u.test(tokens[i + 2]) || tokens[i + 3] !== ';' || bindings.some(binding => binding.key === tokens[i])) return null;
    bindings.push({ key: tokens[i], value: tokens[i + 2] });
  }
  const query = tokens.at(-2)!;
  return bindings.some(binding => binding.key === query) ? { bindings, query } : null;
}

interface Props { choice: ModelChoice; text: string; disabled: boolean; busy: boolean; draft: boolean; onChange: (text: string) => void; onSubmit: (event: FormEvent) => void }
export function InputEditor(props: Props) {
  const [rawOpen, setRawOpen] = useState(false);
  const task = props.choice === 'retrieval' ? retrievalInput(props.text) : null;
  function edit(bindings: Binding[], query: string) {
    props.onChange(`${bindings.map(binding => `${binding.key} = ${binding.value} ;`).join(' ')} ? ${query} =`);
  }
  return <form className="input-editor" onSubmit={props.onSubmit}>
    {task && <div className="binding-editor" role="group" aria-label="Key–value bindings">
      <div className="binding-list">{task.bindings.map((binding, i) => <div className="binding" key={i}>
        <select aria-label={`Binding ${i + 1} key`} value={binding.key} disabled={props.disabled} onChange={event => {
          const key = event.target.value;
          edit(task.bindings.map((item, index) => index === i ? { ...item, key } : item), task.query === binding.key ? key : task.query);
        }}>{['A', 'B', 'C', 'D'].map(key => <option key={key} disabled={task.bindings.some((item, index) => index !== i && item.key === key)}>{key}</option>)}</select>
        <span aria-hidden="true">→</span>
        <select aria-label={`Binding ${i + 1} value`} value={binding.value} disabled={props.disabled} onChange={event => edit(task.bindings.map((item, index) => index === i ? { ...item, value: event.target.value } : item), task.query)}>
          {['0', '1', '2', '3'].map(value => <option key={value}>{value}</option>)}</select>
      </div>)}</div>
      <label className="query-editor">Ask for<select aria-label="Ask for" value={task.query} disabled={props.disabled} onChange={event => edit(task.bindings, event.target.value)}>
        {task.bindings.map(binding => <option key={binding.key}>{binding.key}</option>)}</select></label>
    </div>}
    <div className="raw-input" hidden={!!task && !rawOpen}>
      <label htmlFor="sequence">Space-separated symbols</label>
      <textarea id="sequence" value={props.text} onChange={event => props.onChange(event.target.value)} rows={1} spellCheck={false} disabled={props.disabled} />
      <p className="hint">{props.choice === 'verification' ? 'Use A–O. Up to 7 symbols, plus the beginning token.' : 'Separate every symbol with spaces. Three distinct bindings match the training task. Up to 31 symbols, plus the beginning token.'}</p>
    </div>
    <div className="input-actions">
      {task && <button type="button" className="text-button" aria-expanded={rawOpen} aria-controls="sequence" onClick={() => setRawOpen(!rawOpen)}>{rawOpen ? 'Hide tokens' : 'Edit tokens'}</button>}
      <span className="draft-note">{props.draft ? 'Draft · run to update the input' : task ? `${task.bindings.length} bindings · one question` : 'One next-token prediction'}</span>
      <button type="submit" className="primary" disabled={props.disabled || props.busy}>Run input <span aria-hidden="true">→</span></button>
    </div>
  </form>;
}
