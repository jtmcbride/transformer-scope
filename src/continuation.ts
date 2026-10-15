import { requireInput, type HeadScaleIntervention } from './engine/contracts';
import { logSoftmax } from './engine/comparison';

export interface Sampling { temperature: number; topP: number; seed: number; maxTokens: number }
export interface Choice { tokenId: number; probability: number }
export type Finish = 'end of story' | 'context limit' | 'token limit';
export interface StoryBranch { tokenIds: number[]; choices: Choice[]; finish?: Finish }
export interface StoryFrame { step: number; original: StoryBranch; edited?: StoryBranch }
export type Predict = (tokenIds: number[], edit?: HeadScaleIntervention) => Promise<ArrayLike<number>>;

export interface StorySelection {
  branch: 'original' | 'edited'; step: number; prefix: number[]; choice: Choice; settings: Sampling; edit?: HeadScaleIntervention;
  /** Exact edits for this prediction and subsequent predictions. Absent on legacy, single-head selections. */
  edits?: HeadScaleIntervention[]; continuationEdits?: HeadScaleIntervention[];
}
export function selectionEdits(selection: StorySelection, later = false): HeadScaleIntervention[] {
  return structuredClone(later && selection.continuationEdits !== undefined ? selection.continuationEdits
    : selection.edits ?? (selection.edit ? [selection.edit] : []));
}
/** Reconstruct the input BEFORE the chosen token. Never decode/re-encode a generated prefix. */
export function selectStoryToken(frame: StoryFrame, promptLength: number, branch: StorySelection['branch'], index: number,
  settings: Sampling, edit?: HeadScaleIntervention): StorySelection {
  const story = frame[branch];
  requireInput(!!story && Number.isInteger(index) && index >= 0 && index < story.choices.length, 'Choose a completed generated token.');
  requireInput(Number.isInteger(promptLength) && promptLength > 0 && story!.tokenIds.length === promptLength + story!.choices.length, 'Invalid story prefix.');
  requireInput(branch !== 'edited' || !!edit, 'The edited story needs its captured edit.');
  return structuredClone({ branch, step: index + 1, prefix: story!.tokenIds.slice(0, promptLength + index),
    choice: story!.choices[index], settings, ...(branch === 'edited' ? { edit } : {}) });
}

/** Counter-based randomness gives both branches the same draw at each step, even after divergence. */
export function drawAt(seed: number, step: number): number {
  let value = (seed + Math.imul(step + 1, 0x6D2B79F5)) >>> 0;
  value = Math.imul(value ^ value >>> 15, value | 1);
  value ^= value + Math.imul(value ^ value >>> 7, value | 61);
  return ((value ^ value >>> 14) >>> 0) / 4294967296;
}

/** BOS/PAD are never generated. EOS remains eligible. Probability is from the raw model softmax. */
export function chooseToken(logits: ArrayLike<number>, settings: Sampling, draw: number): Choice {
  requireInput(Number.isFinite(settings.temperature) && settings.temperature >= 0 && settings.temperature <= 2, 'Temperature must be between 0 and 2.');
  requireInput(Number.isFinite(settings.topP) && settings.topP > 0 && settings.topP <= 1, 'Top-p must be greater than 0 and at most 1.');
  requireInput(draw >= 0 && draw < 1, 'Invalid sampling draw.');
  const raw = logSoftmax(logits);
  const ranked = Array.from(raw, (value, tokenId) => ({ tokenId, logit: logits[tokenId], probability: Math.exp(value) }))
    .filter(row => row.tokenId !== 0 && row.tokenId !== 2).sort((a, b) => b.logit - a.logit || a.tokenId - b.tokenId);
  requireInput(ranked.length > 0, 'No eligible next token.');
  if (settings.temperature === 0) return { tokenId: ranked[0].tokenId, probability: ranked[0].probability };
  // Subtract before dividing: very small positive temperatures cannot overflow positive logits.
  const weights = ranked.map(row => Math.exp((row.logit - ranked[0].logit) / settings.temperature));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let mass = 0, end = 0;
  do { mass += weights[end++]; } while (end < weights.length && mass / total < settings.topP);
  let threshold = draw * mass;
  for (let i = 0; i < end; i++) {
    threshold -= weights[i];
    if (threshold < 0 || i === end - 1) return { tokenId: ranked[i].tokenId, probability: ranked[i].probability };
  }
  throw new Error('Could not select a token.');
}

/** Owns exact token prefixes, never re-encodes displayed text. A paired step commits atomically. */
export class StoryContinuation {
  private current: StoryFrame;
  private busy = false;
  readonly settings: Sampling;
  readonly edit?: HeadScaleIntervention;
  constructor(prompt: number[], settings: Sampling, readonly contextLength: number, edit?: HeadScaleIntervention, readonly drawOffset = 0) {
    requireInput(prompt.length > 0 && prompt[0] === 0 && prompt.length <= contextLength, 'Use a completed text prompt within the context limit.');
    requireInput(Number.isInteger(settings.maxTokens) && settings.maxTokens >= 1 && settings.maxTokens <= 64, 'Choose between 1 and 64 new tokens.');
    requireInput(Number.isInteger(settings.seed) && settings.seed >= 0 && settings.seed <= 0xFFFFFFFF, 'Seed must be an integer from 0 to 4294967295.');
    requireInput(Number.isSafeInteger(drawOffset) && drawOffset >= 0, 'Invalid sampling step offset.');
    // Validate sampling settings before creating a worker or retaining any results.
    chooseToken([0, 0, 0, 0], settings, 0);
    requireInput(!edit || edit.kind === 'head.scale' && edit.tokens.kind === 'all' && Number.isFinite(edit.factor), 'Continuation supports one head-output edit at all positions.');
    this.settings = { ...settings }; this.edit = edit ? structuredClone(edit) : undefined;
    const branch = (): StoryBranch => ({ tokenIds: [...prompt], choices: [], ...(prompt.length === contextLength ? { finish: 'context limit' as const } : {}) });
    this.current = { step: 0, original: branch(), ...(edit ? { edited: branch() } : {}) };
  }
  get frame(): StoryFrame { return structuredClone(this.current); }
  get finished(): boolean { return !!this.current.original.finish && (!this.current.edited || !!this.current.edited.finish); }
  async step(predict: Predict): Promise<StoryFrame> {
    requireInput(!this.busy, 'A continuation step is already running.');
    if (this.finished) return this.frame;
    this.busy = true;
    try {
      const draw = drawAt(this.settings.seed, this.drawOffset + this.current.step);
      const advance = async (previous: StoryBranch, edit?: HeadScaleIntervention): Promise<StoryBranch> => {
        if (previous.finish) return previous;
        const choice = chooseToken(await predict([...previous.tokenIds], edit), this.settings, draw);
        const tokenIds = [...previous.tokenIds, choice.tokenId], choices = [...previous.choices, choice];
        const finish: Finish | undefined = choice.tokenId === 1 ? 'end of story' : tokenIds.length >= this.contextLength ? 'context limit'
          : choices.length >= this.settings.maxTokens ? 'token limit' : undefined;
        return { tokenIds, choices, ...(finish ? { finish } : {}) };
      };
      const original = await advance(this.current.original);
      const edited = this.current.edited ? await advance(this.current.edited, this.edit) : undefined;
      this.current = { step: this.current.step + 1, original, ...(edited ? { edited } : {}) };
      return this.frame;
    } finally { this.busy = false; }
  }
}
