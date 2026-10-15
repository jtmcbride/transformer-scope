import { useEffect, useMemo, useRef, useState } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { HeadScaleIntervention, ModelManifest } from '../engine/contracts';
import { tokenLabel } from '../engine/tokenizer';
import { StoryContinuation as Session, selectStoryToken, type Sampling, type StoryBranch, type StoryFrame, type StorySelection } from '../continuation';
import { ModelClient } from '../workers/client';
import { loadStoryModel, predictStory } from '../workers/story-runtime';
import { StoryTokenInspector } from './StoryTokenInspector';

interface Props { manifest: ModelManifest; tokenizer: BpeDocument; tokens: number[]; prompt: string; edit?: HeadScaleIntervention; disabled: boolean; draft: boolean }
const defaults: Sampling = { temperature: 0, topP: .9, seed: 42, maxTokens: 24 };
export function StoryContinuation(props: Props) {
  const [settings, setSettings] = useState(defaults);
  const [compare, setCompare] = useState(false);
  const [frame, setFrame] = useState<StoryFrame | null>(null);
  const [selection, setSelection] = useState<StorySelection | null>(null);
  const [busy, setBusy] = useState(false), [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const session = useRef<Session | null>(null), worker = useRef<ModelClient | null>(null);
  const loaded = useRef<ModelClient | null>(null);
  const epoch = useRef(0), flight = useRef(false), autoplay = useRef(false);
  const encoder = useMemo(() => new BpeTokenizer(props.tokenizer), [props.tokenizer]);
  function stop() {
    ++epoch.current; autoplay.current = false; flight.current = false;
    worker.current?.dispose(); worker.current = null; setBusy(false); setPlaying(false);
  }
  useEffect(() => () => { ++epoch.current; autoplay.current = false; worker.current?.dispose(); }, []);
  function reset() { stop(); session.current = null; setFrame(null); setSelection(null); setError(null); }
  function inspect(branch: StorySelection['branch'], index: number) {
    if (!frame || !session.current) return;
    stop();
    if (selection?.branch === branch && selection.step === index + 1) return;
    setSelection(selectStoryToken(frame, props.tokens.length, branch, index, session.current.settings, session.current.edit));
  }

  async function generate(play: boolean) {
    if (flight.current || props.disabled || session.current?.finished) return;
    setError(null);
    let current = session.current;
    try {
      if (!current) {
        current = new Session(props.tokens, settings, props.manifest.config.contextLength, compare ? props.edit : undefined);
        session.current = current; setFrame(current.frame);
      }
      if (current.finished) return;
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); return; }
    const active = current, generation = ++epoch.current;
    flight.current = true; autoplay.current = play; setBusy(true); setPlaying(play);
    const owner = worker.current ?? new ModelClient(); worker.current = owner;
    const valid = () => epoch.current === generation;
    try {
      // A separate lazy CPU worker keeps generation cancelable and independent of diagram experiments.
      if (owner !== loaded.current) {
        await loadStoryModel(owner, props.manifest);
        if (!valid()) return;
        loaded.current = owner;
      }
      const predict = (tokenIds: number[], edit?: HeadScaleIntervention) => predictStory(owner, tokenIds, edit ? [edit] : []);
      do {
        const next = await active.step(predict);
        if (!valid()) return;
        setFrame(next);
        if (!autoplay.current || active.finished) break;
        // A readable pace; the next step never starts before the current pair completes.
        await new Promise<void>(resolve => setTimeout(resolve, 300));
      } while (valid() && autoplay.current);
    } catch (failure) {
      if (valid()) { setError(failure instanceof Error ? failure.message : String(failure)); owner.dispose(); worker.current = null; }
    } finally {
      if (valid()) { flight.current = false; autoplay.current = false; setBusy(false); setPlaying(false); }
    }
  }
  const finished = !!frame?.original.finish && (!frame.edited || !!frame.edited.finish);
  const frozen = !!frame;
  const captured = session.current?.edit;
  const editLabel = (edit: HeadScaleIntervention) => `Layer ${edit.layer + 1} · Head ${edit.head + 1} · ${edit.factor.toFixed(2)}× at all positions`;
  function story(branch: StoryBranch, name: string) {
    const generated = branch.tokenIds.slice(props.tokens.length);
    const fullText = encoder.preview(branch.tokenIds);
    const last = branch.choices.at(-1);
    return <article className="continuation-story" aria-label={`${name} story`}>
      <div className="continuation-story-heading"><h3>{name}</h3><span>{branch.finish ? `Stopped · ${branch.finish}` : `${generated.length} new tokens`}</span></div>
      <p className="story-prose" data-testid={`${name.toLowerCase()}-story-text`}><span>{props.prompt}</span><strong>{fullText.slice(props.prompt.length)}</strong><i aria-hidden="true" /></p>
      <div className="generated-tokens" aria-label={`${name} generated tokens`}>{generated.map((id, i) => <button type="button" key={i}
        aria-label={`Inspect ${name.toLowerCase()} token ${i + 1}: ${tokenLabel(props.tokenizer, id)}`} aria-pressed={selection?.branch === name.toLowerCase() && selection.step === i + 1}
        disabled={props.disabled} onClick={() => inspect(name === 'Original' ? 'original' : 'edited', i)}
        title={`Inspect step ${i + 1} · token ${id} · model probability ${(branch.choices[i].probability * 100).toFixed(2)}%`}><small>{i + 1}</small>{tokenLabel(props.tokenizer, id)}</button>)}</div>
      {last && <p className="hint">Latest token: <b>{tokenLabel(props.tokenizer, last.tokenId)}</b> · model probability {(last.probability * 100).toFixed(2)}%</p>}
    </article>;
  }
  function download() {
    if (!frame || !session.current) return;
    const data = { format: 'transformer-microscope-continuation', schemaVersion: 1, checkpoint: { modelId: props.manifest.modelId, modelRevision: props.manifest.modelRevision, weightsSha256: props.manifest.weightsSha256, tokenizerSha256: props.manifest.tokenizerSha256 },
      backend: 'cpu', prompt: { text: props.prompt, tokenIds: props.tokens }, settings: session.current.settings, edit: captured ?? null, frame };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'story-continuation.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  return <section id="story-continuation" className="story-continuation" aria-label="Token by token story continuation">
    <div className="continuation-heading"><div><p className="eyebrow">One prediction becomes the next input</p><h2>Watch the story grow.</h2></div><p>Step through one token, or play a short continuation. Each token feeds the next prediction.</p></div>
    <div className="continuation-settings">
      <label>Choose words<select aria-label="Continuation sampling" disabled={frozen || busy} value={settings.temperature === 0 ? 'greedy' : 'sample'} onChange={event => setSettings({ ...settings, temperature: event.target.value === 'greedy' ? 0 : .8 })}><option value="greedy">Most likely token</option><option value="sample">Seeded sampling</option></select></label>
      {settings.temperature > 0 && <><label>Temperature<input aria-label="Continuation temperature" type="number" min="0.05" max="2" step="0.05" disabled={frozen || busy} value={settings.temperature} onChange={event => setSettings({ ...settings, temperature: Number(event.target.value) })} /></label>
        <label>Top-p<input aria-label="Continuation top-p" type="number" min="0.01" max="1" step="0.01" disabled={frozen || busy} value={settings.topP} onChange={event => setSettings({ ...settings, topP: Number(event.target.value) })} /></label>
        <label>Seed<input aria-label="Continuation seed" type="number" min="0" max="4294967295" step="1" disabled={frozen || busy} value={settings.seed} onChange={event => setSettings({ ...settings, seed: Number(event.target.value) })} /></label></>}
      <label>New-token limit<input aria-label="New-token limit" type="number" min="1" max="64" step="1" disabled={frozen || busy} value={settings.maxTokens} onChange={event => setSettings({ ...settings, maxTokens: Number(event.target.value) })} /></label>
      <label className="continuation-compare"><input type="checkbox" checked={frozen ? !!frame.edited : compare && !!props.edit} disabled={frozen || busy || !props.edit} onChange={event => setCompare(event.target.checked)} />Compare with current head edit</label>
    </div>
    {settings.temperature > 0 && <p className="hint">Temperature controls variety. Top-p keeps the most likely choices; the seed makes a run repeatable with the same settings.</p>}
    <p className="hint">{captured ? `Captured edit: ${editLabel(captured)}. Reset to capture another edit.` : compare && props.edit ? `Will compare: ${editLabel(props.edit)}.` : 'Original model. For a comparison, edit a head at all positions, then enable Compare.'} Both versions use the same sampling draw at each step and follow their own generated text.</p>
    <div className="continuation-actions"><button type="button" onClick={() => void generate(false)} disabled={busy || props.disabled || finished}>Step one token</button><button type="button" className="primary" onClick={() => void generate(true)} disabled={busy || props.disabled || finished}>Play story</button><button type="button" onClick={stop} disabled={!busy}>Stop generation</button><button type="button" onClick={reset} disabled={!frame && !busy}>Reset story</button>{frame && <button type="button" onClick={download} disabled={busy}>Download continuation</button>}</div>
    <p className="continuation-status" role="status">{busy ? playing ? 'Generating · one token at a time…' : 'Computing the next token…' : finished ? 'Continuation complete.' : frame ? `Paused after ${frame.step} ${frame.step === 1 ? 'step' : 'steps'}. Step or play to continue.` : 'Ready to continue the completed prompt.'}</p>
    {props.draft && <p className="hint">Your prompt has unrun edits. Generation uses the last completed prompt shown below; Run input starts a new story.</p>}
    {error && <p role="alert" className="error">{error}</p>}
    <div className={`continuation-stories ${frame?.edited ? 'paired-stories' : ''}`}>{story(frame?.original ?? { tokenIds: props.tokens, choices: [] }, 'Original')}{frame?.edited && story(frame.edited, 'Edited')}</div>
    {!!frame?.original.choices.length && <p className="hint">Select a numbered token to pause generation and inspect the computation that produced it.</p>}
    {selection && <StoryTokenInspector key={`${selection.branch}:${selection.step}`} selection={selection} manifest={props.manifest} tokenizer={props.tokenizer}
      onForkStart={stop}
      onClose={() => { setSelection(null); document.querySelector<HTMLButtonElement>('.generated-tokens button[aria-pressed=true]')?.focus({ preventScroll: true }); }} onNavigate={index => inspect(selection.branch, index)}
      count={frame?.[selection.branch]?.choices.length ?? 0} />}
    <p className="hint">Stops at end-of-story, your new-token limit, or {props.manifest.config.contextLength} total tokens including the beginning token. No text is silently dropped. Tokens can be word pieces; incomplete characters wait for more bytes. Downloaded continuations are records, separate from replayable diagram experiments.</p>
  </section>;
}
