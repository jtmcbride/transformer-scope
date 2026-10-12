/** Educational session record (docs/educational-evaluation.md). Contains no names or contact details. */

export type Direction = 'up' | 'down' | 'same';
export type Magnitude = 'none' | 'tiny' | 'moderate' | 'large';
export type AttentionAnswer = 'yes' | 'no' | 'unsure';
export const experienceBands = ['Uses LLMs, no ML course', 'Took an ML course', 'ML practitioner', 'Other'] as const;
export type ExperienceBand = typeof experienceBands[number];

export interface TaskDefinition {
  recipeId: 'effect' | 'small' | 'unreachable'; label: string;
  input: string; predictionPosition: number; answer: string;
  /** As shown in the UI: layers and heads count from 1, positions from 0. */
  layer: number; head: number; positions: 'all' | number; strength: number;
}
export const tasks: readonly TaskDefinition[] = [
  { recipeId: 'effect', label: 'Task 1', input: 'A = 0 ; B = 3 ; C = 0 ; ? B =', predictionPosition: 15, answer: '3', layer: 1, head: 4, positions: 'all', strength: 0 },
  { recipeId: 'small', label: 'Task 2', input: 'A = 2 ; C = 0 ; B = 3 ; ? A =', predictionPosition: 15, answer: '2', layer: 1, head: 2, positions: 0, strength: 0 },
  { recipeId: 'unreachable', label: 'Task 3', input: 'A = 0 ; B = 3 ; C = 0 ; ? B =', predictionPosition: 15, answer: '3', layer: 2, head: 1, positions: 0, strength: 0 },
];
export const probes = [
  { id: 'C1', prompt: 'When you set Head 4 in Layer 1 to 0 in Task 1, what happened to that head\'s attention weights? Why? (Check one attention cell at strength 1 and at strength 0.)' },
  { id: 'C2', prompt: 'In Task 1 the head\'s output changed. List what else in the model changed as a result, in order, up to the final prediction.' },
  { id: 'C3', prompt: 'Tasks 2 and 3 both showed little or no change. Are they the same kind of result? Could either show a larger change for a different input?' },
  { id: 'C4', prompt: 'Based on Task 1, what does Layer 1 Head 4 do? How confident are you, and what would you need to check?' },
  { id: 'C5', prompt: 'Answer before running: on the Task 1 input, could editing Layer 2, Head 1 at position 15 only change the prediction after position 15? What about at position 14 only? Then check in the tool.' },
] as const;
export type ProbeId = typeof probes[number]['id'];

export interface Prediction { direction: Direction; magnitude: Magnitude; reasoning: string; attention: AttentionAnswer; attentionReason: string; recordedAt: string }
/** Values read from the completed result the participant saw, before any further interaction. */
export interface Observation {
  recipeTitle: string; targetSymbol: string; baselinePercent: number; modifiedPercent: number;
  /** Displayed text, kept because a rounded change such as "-0.0000" still carries its sign. */
  logProbabilityDeltaText: string; jsDivergence: number; noPathNote: boolean; revealedAt: string;
}
export interface TaskRecord { recipeId: TaskDefinition['recipeId']; prediction: Prediction; observation: Observation; followUp: { match: string; explanation: string } }
export interface SessionDocument {
  format: 'transformer-microscope-study-session'; schemaVersion: 1;
  release: { commit: string | null; modelRevision: string | null };
  participant: { id: string; experienceBand: ExperienceBand; device: string };
  consent: true; startedAt: string; completedAt: string;
  orientationNotes: string; tasks: TaskRecord[];
  probes: Record<ProbeId, { answer: string; hint: boolean }>; debrief: string;
}

export const participantPattern = /^P\d{2,3}$/u;

/** Mechanical observed category. No change means an exactly zero distribution change (JSD 0). */
export function observedDirection(observation: Pick<Observation, 'jsDivergence' | 'logProbabilityDeltaText'>): Direction {
  if (observation.jsDivergence === 0) return 'same';
  const text = observation.logProbabilityDeltaText.trim().replace(/\u2212/gu, '-');
  return text.startsWith('-') ? 'down' : displayedNumber(text) > 0 ? 'up' : 'same';
}
/** Bands from the prediction card: tiny is a non-zero change under 1 percentage point; large is over 50. */
export function observedMagnitude(observation: Pick<Observation, 'jsDivergence' | 'baselinePercent' | 'modifiedPercent'>): Magnitude {
  if (observation.jsDivergence === 0) return 'none';
  const change = Math.abs(observation.modifiedPercent - observation.baselinePercent);
  return change < 1 ? 'tiny' : change > 50 ? 'large' : 'moderate';
}
export function scoreTask(record: Pick<TaskRecord, 'prediction' | 'observation'>) {
  return { direction: record.prediction.direction === observedDirection(record.observation),
    magnitude: record.prediction.magnitude === observedMagnitude(record.observation),
    attentionUnchanged: record.prediction.attention === 'no' };
}

/** Parses a displayed number such as "99.99%", "-0.0000", "8.750e-12 nats", or a Unicode minus. */
export function displayedNumber(text: string): number {
  const match = text.replace(/−/gu, '-').match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/iu);
  if (!match) throw new Error(`No number in "${text}".`);
  return Number(match[0]);
}

export function validateSession(value: unknown): SessionDocument {
  const session = value as SessionDocument;
  const fail = (reason: string): never => { throw new Error(`Invalid session record: ${reason}.`); };
  if (session?.format !== 'transformer-microscope-study-session' || session.schemaVersion !== 1) fail('unsupported format');
  if (!participantPattern.test(session.participant?.id ?? '')) fail('participant ID must look like P01');
  if (!experienceBands.includes(session.participant.experienceBand)) fail('unknown experience band');
  if (session.consent !== true) fail('consent was not recorded');
  if (!Array.isArray(session.tasks) || session.tasks.map(task => task.recipeId).join() !== tasks.map(task => task.recipeId).join()) fail('tasks must be the three recipes in order');
  for (const task of session.tasks) {
    const { prediction, observation } = task;
    if (!['up', 'down', 'same'].includes(prediction?.direction) || !['none', 'tiny', 'moderate', 'large'].includes(prediction.magnitude) || !['yes', 'no', 'unsure'].includes(prediction.attention)) fail(`${task.recipeId} prediction is incomplete`);
    if (Date.parse(prediction.recordedAt) > Date.parse(observation?.revealedAt)) fail(`${task.recipeId} prediction was recorded after the result`);
    if (![observation.baselinePercent, observation.modifiedPercent, observation.jsDivergence].every(Number.isFinite) || !Number.isFinite(displayedNumber(observation.logProbabilityDeltaText))) fail(`${task.recipeId} observation is not numeric`);
  }
  for (const probe of probes) if (typeof session.probes?.[probe.id]?.answer !== 'string' || typeof session.probes[probe.id].hint !== 'boolean') fail(`${probe.id} is missing`);
  return session;
}
