import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import type { ExperimentRecipe } from '../../src/experiments';
import { displayedNumber, observedDirection, observedMagnitude, scoreTask, tasks, validateSession, type Observation, type SessionDocument } from '../../src/study/session';

const symbols = (JSON.parse(readFileSync('public/models/retrieval/tokenizer.json', 'utf8')) as { tokens: string[] }).tokens;

it('prediction cards state exactly the shipped recipes in UI numbering', () => {
  const recipes = (JSON.parse(readFileSync('public/models/retrieval/recipes.json', 'utf8')) as { recipes: ExperimentRecipe[] }).recipes;
  expect(tasks.map(task => task.recipeId)).toEqual(recipes.map(recipe => recipe.id));
  for (const task of tasks) {
    const recipe = recipes.find(item => item.id === task.recipeId)!;
    const { intervention } = recipe;
    expect(task.input).toBe(recipe.text);
    expect(task.predictionPosition).toBe(recipe.predictionPosition);
    expect(task.answer).toBe(symbols[recipe.targetTokenId]);
    expect([task.layer, task.head, task.strength]).toEqual([intervention.layer + 1, intervention.head + 1, intervention.factor]);
    expect(task.positions).toEqual(intervention.tokens.kind === 'all' ? 'all' : intervention.tokens.indices[0]);
  }
});

const seen = (patch: Partial<Observation>): Observation => ({ recipeTitle: '', targetSymbol: '3', baselinePercent: 99.99, modifiedPercent: 99.99,
  logProbabilityDeltaText: '0.0000', jsDivergence: 0, noPathNote: false, revealedAt: '2026-10-12T00:00:01Z', ...patch });

it('categorizes displayed results, keeping the sign of a rounded change', () => {
  // Displayed values for the three recipes.
  const effect = seen({ modifiedPercent: 0, logProbabilityDeltaText: '-11.7770', jsDivergence: .6928 });
  const small = seen({ logProbabilityDeltaText: '-0.0000', jsDivergence: 8.75e-12 });
  const none = seen({ noPathNote: true });
  expect([observedDirection(effect), observedMagnitude(effect)]).toEqual(['down', 'large']);
  expect([observedDirection(small), observedMagnitude(small)]).toEqual(['down', 'tiny']);
  expect([observedDirection(none), observedMagnitude(none)]).toEqual(['same', 'none']);
  expect(observedDirection(seen({ logProbabilityDeltaText: '0.2500', jsDivergence: .01 }))).toBe('up');
  expect(observedDirection(seen({ logProbabilityDeltaText: '−0.1000', jsDivergence: .01 }))).toBe('down');
  expect(observedMagnitude(seen({ modifiedPercent: 70, jsDivergence: .1 }))).toBe('moderate');
  expect(observedMagnitude(seen({ modifiedPercent: 49.99, jsDivergence: .1 }))).toBe('moderate');
  expect(observedMagnitude(seen({ modifiedPercent: 49.98, jsDivergence: .1 }))).toBe('large');
  expect(displayedNumber('8.750e-12 nats')).toBe(8.75e-12);
  expect(displayedNumber('99.99%')).toBe(99.99);
  expect(() => displayedNumber('n/a')).toThrow();
});

it('scores predictions mechanically and validates complete session records', () => {
  const prediction = { direction: 'down' as const, magnitude: 'large' as const, reasoning: 'r', attention: 'no' as const, attentionReason: '', recordedAt: '2026-10-12T00:00:00Z' };
  const effect = seen({ modifiedPercent: 0, logProbabilityDeltaText: '-11.7770', jsDivergence: .69 });
  expect(scoreTask({ prediction, observation: effect })).toEqual({ direction: true, magnitude: true, attentionUnchanged: true });
  expect(scoreTask({ prediction: { ...prediction, attention: 'unsure' }, observation: seen({}) })).toEqual({ direction: false, magnitude: false, attentionUnchanged: false });
  const session: SessionDocument = { format: 'transformer-microscope-study-session', schemaVersion: 1, release: { commit: null, modelRevision: null },
    participant: { id: 'P01', experienceBand: 'Took an ML course', device: '' }, consent: true, startedAt: '', completedAt: '', orientationNotes: '',
    tasks: tasks.map(task => ({ recipeId: task.recipeId, prediction, observation: effect, followUp: { match: '', explanation: '' } })),
    probes: { C1: { answer: '', hint: false }, C2: { answer: '', hint: false }, C3: { answer: '', hint: true }, C4: { answer: '', hint: false }, C5: { answer: '', hint: false } }, debrief: '' };
  expect(validateSession(structuredClone(session))).toEqual(session);
  expect(() => validateSession({ ...session, participant: { ...session.participant, id: 'Alice' } })).toThrow('P01');
  expect(() => validateSession({ ...session, consent: false })).toThrow('consent');
  expect(() => validateSession({ ...session, tasks: session.tasks.slice(1) })).toThrow('three recipes');
  const late = structuredClone(session); late.tasks[0].prediction.recordedAt = '2026-10-12T00:00:05Z';
  expect(() => validateSession(late)).toThrow('after the result');
  const missing = structuredClone(session) as unknown as { probes: Record<string, unknown> }; delete missing.probes.C5;
  expect(() => validateSession(missing)).toThrow('C5');
});

it('summarizes sessions into findings tables and evaluates the gate only from human codes', async () => {
  const { summarizeStudy } = await import('../../scripts/summarize_study.mjs');
  const observations: Record<string, Observation> = {
    effect: seen({ modifiedPercent: 0, logProbabilityDeltaText: '-11.7770', jsDivergence: .69 }),
    small: seen({ logProbabilityDeltaText: '-0.0000', jsDivergence: 8.75e-12 }),
    unreachable: seen({ noPathNote: true }),
  };
  // Synthetic records for exercising the script; they are not participant data.
  const session = (id: string, hint = false): SessionDocument => ({ format: 'transformer-microscope-study-session', schemaVersion: 1,
    release: { commit: 'a'.repeat(40), modelRevision: null }, participant: { id, experienceBand: 'Took an ML course', device: 'test' }, consent: true,
    startedAt: '2026-10-12T00:00:00Z', completedAt: '2026-10-12T00:30:00Z', orientationNotes: '',
    tasks: tasks.map(task => ({ recipeId: task.recipeId, observation: observations[task.recipeId], followUp: { match: 'm', explanation: '' },
      prediction: { direction: 'down', magnitude: 'large', reasoning: 'a | b', attention: 'no', attentionReason: '', recordedAt: '2026-10-12T00:00:00Z' } })),
    probes: { C1: { answer: 'unchanged', hint }, C2: { answer: 'x', hint: false }, C3: { answer: 'x', hint: false }, C4: { answer: 'x', hint: false }, C5: { answer: 'x', hint: false } }, debrief: '' });
  const ids = ['P01', 'P02', 'P03', 'P04', 'P05'];
  const all = { C1: 'C', C2: 'C', C3: 'C', C4: 'P', C5: 'C' };
  const codes = Object.fromEntries(ids.map(id => [id, { ...all }]));
  codes.P05.C2 = 'M';
  const met = summarizeStudy([...ids.slice(1).map(id => session(id)), session('P01', true)], codes);
  expect(met).toContain('| C1 correct without a hint | ≥ 4 of 5 (and n ≥ 5) | 4 of 5 (1 more after a hint) | yes |');
  expect(met).toContain('| C2 correct without a hint | ≥ 4 of 5 (and n ≥ 5) | 4 of 5 (0 more after a hint) | yes |');
  expect(met).toContain('| C2 | P05 |');
  expect(met).toContain('| Task 1 (effect) direction predicted correctly | reported only | 5 of 5 | n/a |');
  expect(met).toContain('| Task 3 (unreachable) direction predicted correctly | reported only | 0 of 5 | n/a |');
  expect(met).toContain('a \\| b');
  expect(met.indexOf('| P01 |')).toBeLessThan(met.indexOf('| P02 |'));
  codes.P04.C1 = 'P';
  expect(summarizeStudy([...ids.slice(1).map(id => session(id)), session('P01', true)], codes)).toContain('| C1 correct without a hint | ≥ 4 of 5 (and n ≥ 5) | 3 of 5 (1 more after a hint) | no |');
  const uncoded = summarizeStudy([session('P01'), session('P02')]);
  expect(uncoded).toContain('not evaluated: codes missing');
  expect(uncoded).toContain('Only 2 session(s)');
  expect(() => summarizeStudy([session('P01'), session('P01')])).toThrow('once');
  expect(() => summarizeStudy([session('P01')], { P01: { C1: 'yes' } })).toThrow('Invalid code');
  expect(() => summarizeStudy([session('P01')], { P09: { C1: 'C' } })).toThrow('unknown participant');
});
