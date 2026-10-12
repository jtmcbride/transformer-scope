import { useEffect, useState } from 'react';

export function StrengthNumber({ value, disabled, onCommit }: { value: number; disabled: boolean; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState(String(value));
  const [invalid, setInvalid] = useState(false);
  useEffect(() => { setDraft(String(value)); setInvalid(false); }, [value]);
  function commit() {
    const next = Number(draft);
    if (!draft.trim() || !Number.isFinite(next) || next < 0 || next > 2) { setInvalid(true); return; }
    setInvalid(false); if (next !== value) onCommit(next);
  }
  return <div className="numeric-strength"><label>Exact head strength<input type="number" min="0" max="2" step="any" value={draft} disabled={disabled}
    aria-invalid={invalid} aria-describedby={invalid ? 'strength-error' : undefined}
    onChange={event => { setDraft(event.target.value); setInvalid(false); }} onBlur={commit}
    onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); commit(); } }} /></label>
    {invalid && <p id="strength-error" className="field-error" role="alert">Use a strength from 0 to 2.</p>}
  </div>;
}
