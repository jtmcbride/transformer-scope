import { useEffect, useRef, useState } from 'react';
import { button, change, openFrame, select, settled, statusText, waitFor, type Frame } from '../embedded/frame';
import {
  displayedNumber, experienceBands, observedDirection, observedMagnitude, participantPattern, probes, scoreTask, tasks, validateSession,
  type AttentionAnswer, type Direction, type ExperienceBand, type Magnitude, type Observation, type Prediction, type ProbeId, type SessionDocument, type TaskRecord,
} from './session';

const draftKey = 'transformer-microscope-study-draft';
type Step = 'setup' | 'intro' | 'orientation' | 'task' | 'probes' | 'debrief' | 'done';
interface Draft {
  step: Step; taskIndex: number; startedAt: string; participant: { id: string; experienceBand: ExperienceBand | ''; device: string }; consent: boolean;
  orientationNotes: string; predictions: (Partial<Prediction> | undefined)[]; observations: (Observation | undefined)[];
  followUps: { match: string; explanation: string }[]; probes: Record<ProbeId, { answer: string; hint: boolean }>; debrief: string;
}
const blank = (): Draft => ({ step: 'setup', taskIndex: 0, startedAt: '', participant: { id: '', experienceBand: '', device: '' }, consent: false,
  orientationNotes: '', predictions: [], observations: [], followUps: tasks.map(() => ({ match: '', explanation: '' })),
  probes: Object.fromEntries(probes.map(probe => [probe.id, { answer: '', hint: false }])) as Draft['probes'], debrief: '' });
function readDraft(): Draft | null {
  try { const text = localStorage.getItem(draftKey); return text ? JSON.parse(text) as Draft : null; } catch { return null; }
}
const base = new URL('./', window.location.href);
export const orientationInput = 'A = 1 ; B = 2 ; C = 3 ; ? C =';

/** The application opens on Task 1's recipe; orientation must show a neutral baseline instead, before the frame is revealed. */
async function presetOrientation(frame: Frame) {
  const textarea = frame.doc.getElementById('sequence') as HTMLTextAreaElement;
  Object.getOwnPropertyDescriptor(frame.win.HTMLTextAreaElement.prototype, 'value')!.set!.call(textarea, orientationInput);
  textarea.dispatchEvent(new frame.win.Event('input', { bubbles: true }));
  const loading = waitFor(frame, () => statusText(frame.doc).startsWith('Loading'), 'the orientation input to start');
  textarea.form!.requestSubmit();
  await loading; await waitFor(frame, () => settled(frame.doc), 'the orientation input');
  const computing = waitFor(frame, () => statusText(frame.doc).startsWith('Computing'), 'the baseline to start');
  button(frame.doc, 'Reset to baseline')!.click();
  await computing; await waitFor(frame, () => settled(frame.doc) && statusText(frame.doc) === 'Showing 1.00×', 'the orientation baseline');
}

function observe(frame: Frame, answer: string): Observation {
  const doc = frame.doc;
  const row = [...doc.querySelectorAll('.probability-row')].find(item => item.querySelector('.symbol')?.textContent === answer);
  const metrics = doc.querySelectorAll('.metrics strong');
  if (!row || metrics.length < 2) throw new Error('The completed result is not visible.');
  return { recipeTitle: doc.querySelector('[data-testid="recipe-note"] strong')?.textContent ?? '', targetSymbol: answer,
    baselinePercent: displayedNumber(row.querySelector('.numbers span')!.textContent!), modifiedPercent: displayedNumber(row.querySelector('.numbers strong')!.textContent!),
    jsDivergence: displayedNumber(metrics[0].textContent!), logProbabilityDeltaText: metrics[1].firstChild?.textContent?.trim() ?? '',
    noPathNote: !!doc.querySelector('.path-note'), revealedAt: new Date().toISOString() };
}

const introduction = 'This tool runs a small trained transformer in your browser. It was trained on one task: read three key–value pairs, such as A = 0 ; B = 3 ; C = 1, then answer a query such as ? A = with that key\'s value. Each layer contains several attention heads. Each head computes attention weights over earlier positions, combines information using those weights, and adds its output to the computation. You can change how strongly one head\'s output is added: 1 is normal, 0 removes it, and 2 doubles it. We are testing the tool, not you. Please think aloud.';

export default function StudyApp() {
  const host = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<Draft>(blank);
  const [saved] = useState(readDraft);
  const [resumeOffered, setResumeOffered] = useState(() => !!saved && saved.step !== 'setup');
  const [frameState, setFrameState] = useState<'none' | 'loading' | 'ready'>('none');
  const [error, setError] = useState<string | null>(null);
  const [release, setRelease] = useState<{ commit: string | null; modelRevision: string | null }>({ commit: null, modelRevision: null });
  const [exported, setExported] = useState<SessionDocument | null>(null);
  const task = tasks[draft.taskIndex];
  const prediction = draft.predictions[draft.taskIndex] ?? {};
  const observation = draft.observations[draft.taskIndex];

  useEffect(() => {
    void fetch(new URL('release.json', base), { cache: 'no-store' }).then(response => response.ok ? response.json() : null)
      .then(stamp => setRelease({ commit: stamp?.commit ?? null, modelRevision: stamp?.models?.retrieval?.modelRevision ?? null })).catch(() => {});
  }, []);
  useEffect(() => {
    if (draft.step === 'setup' || draft.step === 'done') return;
    try { localStorage.setItem(draftKey, JSON.stringify(draft)); } catch { /* A draft is a convenience; the export is the record. */ }
  }, [draft]);
  // Loads the application for exploration steps, and for a task only after its prediction is recorded.
  useEffect(() => {
    const explore = draft.step === 'orientation' || draft.step === 'probes';
    const reveal = draft.step === 'task' && !!prediction.recordedAt;
    if (!explore && !reveal) { host.current?.replaceChildren(); setFrameState('none'); return; }
    let current = true;
    setFrameState('loading'); setError(null);
    void (async () => {
      const frame = await openFrame(host.current!, base, 'Transformer Microscope', 'study-frame');
      await waitFor(frame, () => settled(frame.doc), 'the application to load');
      if (draft.step === 'orientation') await presetOrientation(frame);
      if (reveal) {
        await change(frame, select(frame.doc, 'Prepared experiment'), task.recipeId, 'Loading');
        if (select(frame.doc, 'Prepared experiment').value !== task.recipeId) throw new Error('The prepared experiment did not load.');
        if (!current) return;
        if (!draft.observations[draft.taskIndex]) {
          const seen = observe(frame, task.answer);
          setDraft(previous => { const observations = [...previous.observations]; observations[previous.taskIndex] = seen; return { ...previous, observations }; });
        }
      }
      if (current) setFrameState('ready');
    })().catch(failure => { if (current) setError(failure instanceof Error ? failure.message : String(failure)); });
    return () => { current = false; };
  }, [draft.step, draft.taskIndex, !!prediction.recordedAt]);

  const update = (patch: Partial<Draft>) => setDraft(previous => ({ ...previous, ...patch }));
  const predict = (patch: Partial<Prediction>) => setDraft(previous => {
    const predictions = [...previous.predictions]; predictions[previous.taskIndex] = { ...predictions[previous.taskIndex], ...patch }; return { ...previous, predictions };
  });
  const follow = (patch: Partial<Draft['followUps'][number]>) => setDraft(previous => {
    const followUps = [...previous.followUps]; followUps[previous.taskIndex] = { ...followUps[previous.taskIndex], ...patch }; return { ...previous, followUps };
  });
  const setupValid = participantPattern.test(draft.participant.id) && draft.participant.experienceBand !== '' && draft.consent;
  const predictionComplete = !!prediction.direction && !!prediction.magnitude && !!prediction.attention && !!prediction.reasoning?.trim();

  function finish() {
    try {
      const session = validateSession({
        format: 'transformer-microscope-study-session', schemaVersion: 1, release, participant: { ...draft.participant, device: draft.participant.device.trim() },
        consent: true, startedAt: draft.startedAt, completedAt: new Date().toISOString(), orientationNotes: draft.orientationNotes,
        tasks: tasks.map((definition, index): TaskRecord => ({ recipeId: definition.recipeId, prediction: draft.predictions[index] as Prediction,
          observation: draft.observations[index]!, followUp: draft.followUps[index] })),
        probes: draft.probes, debrief: draft.debrief,
      });
      setExported(session); update({ step: 'done' });
      try { localStorage.removeItem(draftKey); } catch { /* ignored */ }
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
  }
  function download(session: SessionDocument) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(session, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `study-session-${session.participant.id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const choice = <T extends string>(name: string, value: T | undefined, options: readonly [T, string][], onChange: (value: T) => void, locked: boolean) =>
    <div className="choice-row" role="radiogroup" aria-label={name}>{options.map(([option, text]) => <label key={option}>
      <input type="radio" name={name} checked={value === option} disabled={locked} onChange={() => onChange(option)} />{text}</label>)}</div>;
  const locked = !!prediction.recordedAt;

  return <main className="study">
    <header><p className="eyebrow">Facilitated session · not linked from the application</p><h1>Educational evaluation session</h1>
      <p>Protocol: docs/educational-evaluation.md. Records stay on this device until you export them. Do not enter names or contact details.</p></header>
    {error && <p className="error" role="alert">{error}</p>}
    {resumeOffered && saved && <section className="panel"><p>An unfinished session for {saved.participant.id} is saved on this device.</p>
      <div className="actions"><button type="button" onClick={() => { setDraft(saved); setResumeOffered(false); }}>Resume {saved.participant.id}</button>
        <button type="button" onClick={() => { try { localStorage.removeItem(draftKey); } catch { /* ignored */ } setResumeOffered(false); }}>Discard</button></div></section>}

    {draft.step === 'setup' && !resumeOffered && <section className="panel" aria-label="Session setup">
      <label>Participant ID<input aria-label="Participant ID" value={draft.participant.id} placeholder="P01" onChange={event => update({ participant: { ...draft.participant, id: event.target.value.trim().toUpperCase() } })} /></label>
      <label>Experience band<select aria-label="Experience band" value={draft.participant.experienceBand} onChange={event => update({ participant: { ...draft.participant, experienceBand: event.target.value as ExperienceBand } })}>
        <option value="" disabled>Choose…</option>{experienceBands.map(band => <option key={band}>{band}</option>)}</select></label>
      <label>Device and browser<input aria-label="Device and browser" value={draft.participant.device} maxLength={120} placeholder="e.g. iPhone 13, Safari 26" onChange={event => update({ participant: { ...draft.participant, device: event.target.value } })} /></label>
      <label className="inline"><input type="checkbox" checked={draft.consent} onChange={event => update({ consent: event.target.checked })} />The participant gave informed consent and may stop at any time.</label>
      <p className="hint">Deployed commit: {release.commit ? release.commit.slice(0, 7) : 'not identified'}</p>
      <button type="button" className="primary" disabled={!setupValid} onClick={() => update({ step: 'intro', startedAt: new Date().toISOString() })}>Start session</button>
    </section>}

    {draft.step === 'intro' && <section className="panel" aria-label="Introduction"><h2>Introduction · read verbatim</h2><blockquote>{introduction}</blockquote>
      <p className="hint">Answer interface questions, not conceptual ones, until the debrief.</p>
      <button type="button" className="primary" onClick={() => update({ step: 'orientation' })}>Continue to orientation</button></section>}

    {draft.step === 'orientation' && <section className="panel" aria-label="Orientation"><h2>Orientation</h2>
      <p>The application below starts on <code>{orientationInput}</code> at the baseline. Ask the participant to find the predicted answer, select a token, and move the strength slider once.</p>
      <label>Interface difficulties observed<textarea aria-label="Orientation notes" rows={2} value={draft.orientationNotes} onChange={event => update({ orientationNotes: event.target.value })} /></label>
      <button type="button" className="primary" onClick={() => update({ step: 'task', taskIndex: 0 })}>Continue to Task 1</button></section>}

    {draft.step === 'task' && <section className="panel" aria-label={task.label}><h2>{task.label}: prediction card</h2>
      <dl className="card"><div><dt>Input</dt><dd><code>{task.input}</code></dd></div>
        <div><dt>Prediction</dt><dd>After position {task.predictionPosition} (the final <code>=</code>); the correct answer is <code>{task.answer}</code></dd></div>
        <div><dt>Intervention</dt><dd>Layer {task.layer}, Head {task.head}, {task.positions === 'all' ? 'all positions' : `position ${task.positions} (<BOS>) only`}, strength {task.strength}</dd></div></dl>
      <fieldset disabled={locked}><legend>Before seeing the result</legend>
        <p>P1. Will the probability of the correct answer go up, go down, or stay the same?</p>
        {choice<Direction>('P1 direction', prediction.direction, [['up', 'Up'], ['down', 'Down'], ['same', 'Stay the same']], direction => predict({ direction }), locked)}
        <p>P2. How much?</p>
        {choice<Magnitude>('P2 magnitude', prediction.magnitude, [['none', 'No change'], ['tiny', 'Tiny (under 1 point)'], ['moderate', 'Moderate'], ['large', 'Large (over 50 points)']], magnitude => predict({ magnitude }), locked)}
        <label>P3. Why?<textarea aria-label="P3 reasoning" rows={2} value={prediction.reasoning ?? ''} onChange={event => predict({ reasoning: event.target.value })} /></label>
        <p>P4. Will this head's attention weights change?</p>
        {choice<AttentionAnswer>('P4 attention', prediction.attention, [['yes', 'Yes'], ['no', 'No'], ['unsure', 'Unsure']], attention => predict({ attention }), locked)}
        <label>Why?<textarea aria-label="P4 reason" rows={1} value={prediction.attentionReason ?? ''} onChange={event => predict({ attentionReason: event.target.value })} /></label>
      </fieldset>
      {!locked && <button type="button" className="primary" disabled={!predictionComplete} onClick={() => predict({ attentionReason: prediction.attentionReason ?? '', recordedAt: new Date().toISOString() })}>Record prediction and show the result</button>}
      {locked && frameState !== 'ready' && <p role="status">Preparing the result…</p>}
      {observation && frameState === 'ready' && <div className="after" aria-label="After the result">
        <p className="hint" data-testid="observed">Recorded result: {observation.targetSymbol} {observation.baselinePercent.toFixed(2)}% → {observation.modifiedPercent.toFixed(2)}% ·
          {' '}{observedDirection(observation)}, {observedMagnitude(observation)}{observation.noPathNote ? ' · no computational path' : ''}</p>
        <label>R1. Did the result match your prediction? What explains any difference?<textarea aria-label="R1 match" rows={2} value={draft.followUps[draft.taskIndex].match} onChange={event => follow({ match: event.target.value })} /></label>
        {draft.taskIndex > 0 && <label>R2. The prediction barely changed or did not change. Does the tool say why?<textarea aria-label="R2 explanation" rows={2} value={draft.followUps[draft.taskIndex].explanation} onChange={event => follow({ explanation: event.target.value })} /></label>}
        <button type="button" className="primary" onClick={() => update(draft.taskIndex < tasks.length - 1 ? { taskIndex: draft.taskIndex + 1 } : { step: 'probes' })}>
          {draft.taskIndex < tasks.length - 1 ? `Continue to ${tasks[draft.taskIndex + 1].label}` : 'Continue to concept probes'}</button>
      </div>}
    </section>}

    {draft.step === 'probes' && <section className="panel" aria-label="Concept probes"><h2>Concept probes</h2><p className="hint">Record answers verbatim. The participant may use the tool below.</p>
      {probes.map(probe => <div className="probe" key={probe.id}><label>{probe.id}. {probe.prompt}<textarea aria-label={`${probe.id} answer`} rows={2} value={draft.probes[probe.id].answer}
        onChange={event => update({ probes: { ...draft.probes, [probe.id]: { ...draft.probes[probe.id], answer: event.target.value } } })} /></label>
        <label className="inline"><input type="checkbox" aria-label={`${probe.id} hint given`} checked={draft.probes[probe.id].hint}
          onChange={event => update({ probes: { ...draft.probes, [probe.id]: { ...draft.probes[probe.id], hint: event.target.checked } } })} />Facilitator gave a hint</label></div>)}
      <button type="button" className="primary" onClick={() => update({ step: 'debrief' })}>Continue to debrief</button></section>}

    {draft.step === 'debrief' && <section className="panel" aria-label="Debrief"><h2>Debrief</h2>
      <p>Explain the correct answers to any missed probes (see the coding table in the protocol), and ask which part of the screen would have helped. For C4, share the head-ablation coverage in docs/retrieval.md only after the participant has answered.</p>
      <label>Debrief notes and suggestions<textarea aria-label="Debrief notes" rows={3} value={draft.debrief} onChange={event => update({ debrief: event.target.value })} /></label>
      <button type="button" className="primary" onClick={finish}>Finish and export</button></section>}

    {draft.step === 'done' && exported && <section className="panel" aria-label="Session record"><h2>Session {exported.participant.id} recorded</h2>
      <table><thead><tr><th>Task</th><th>Predicted</th><th>Observed</th><th>Direction</th><th>Magnitude</th><th>Attention unchanged</th></tr></thead>
        <tbody>{exported.tasks.map((record, index) => { const score = scoreTask(record); return <tr key={record.recipeId}><td>{tasks[index].label}</td>
          <td>{record.prediction.direction}, {record.prediction.magnitude}</td><td>{observedDirection(record.observation)}, {observedMagnitude(record.observation)}</td>
          <td>{score.direction ? 'correct' : 'incorrect'}</td><td>{score.magnitude ? 'correct' : 'incorrect'}</td><td>{score.attentionUnchanged ? 'correct' : 'incorrect'}</td></tr>; })}</tbody></table>
      <p className="hint">Concept probes are coded by people, not by this page. Summarize sessions with <code>node scripts/summarize_study.mjs</code>.</p>
      <div className="actions"><button type="button" onClick={() => download(exported)}>Download session JSON</button>
        <button type="button" onClick={() => { setExported(null); setDraft(blank()); }}>Start another session</button></div>
      <details><summary>Raw JSON</summary><pre data-testid="session-json">{JSON.stringify(exported, null, 2)}</pre></details></section>}

    <div ref={host} className={`frame-host ${frameState === 'ready' ? '' : 'hidden'}`} />
  </main>;
}
