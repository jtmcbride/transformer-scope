import { requireInput } from './engine/contracts';

export const textExamples = [
  { label: 'Lily’s story', text: 'Once upon a time, there was a little girl named Lily. One day,' },
  { label: 'The red ball', text: 'Tom found a red ball. He wanted to' },
  { label: 'A forest walk', text: 'Ben walked into the forest. He saw a little' },
];
export const defaultStory = textExamples[0].text;
export interface TextEvaluation {
  modelRevision: string; status: 'experimental-language-model'; originalAcceptanceGate: 'failed';
  training: { step: number; validation: { mean: number } }; baselines: { bigram: number; unigram: number };
  nameCopying: { correct: number; count: number; rate: number; threshold: number };
  pronounAgreement: { correct: number; count: number; rate: number };
}
export async function fetchTextEvaluation(manifestUrl: string): Promise<unknown> {
  const response = await fetch(new URL('evaluation.json', manifestUrl));
  requireInput(response.ok, 'Could not load the TinyStories checkpoint report. Refresh to retry.');
  return response.json();
}
export function validateTextEvaluation(value: unknown, revision: string): TextEvaluation {
  const report = value as TextEvaluation;
  requireInput(report?.modelRevision === revision && report.status === 'experimental-language-model' && report.originalAcceptanceGate === 'failed', 'The TinyStories report does not match the checkpoint.');
  requireInput([report.training?.validation?.mean, report.baselines?.bigram, report.baselines?.unigram, report.nameCopying?.rate, report.pronounAgreement?.rate].every(Number.isFinite), 'Invalid TinyStories evaluation measurements.');
  return report;
}
