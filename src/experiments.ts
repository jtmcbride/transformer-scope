import { tokenize, validateInterventions, validateTokens, type HeadScaleIntervention, type ModelManifest, type ModelTokenizer } from './engine/contracts';

export type ModelChoice = 'retrieval' | 'verification' | 'tinystories';
export interface ExperimentRecipe {
  id: string; title: string; interpretation: string; modelRevision: string; text: string;
  tokenIds: number[]; predictionPosition: number; targetTokenId: number; intervention: HeadScaleIntervention;
  metrics: { baselineLogits: number[]; modifiedLogits: number[]; baselineTargetProbability: number;
    modifiedTargetProbability: number; targetLogProbabilityDelta: number; jsDivergence: number; maxAbsoluteLogitChange: number };
}
export interface RetrievalEvaluation {
  modelRevision: string;
  heldOut: { count: number; accuracy: number; lossNats: number };
  localContext: { accuracy: number };
  lengthGeneralization: Record<string, { count: number; accuracy: number }>;
}
export interface ExperimentMetadata { recipes: ExperimentRecipe[]; evaluation: RetrievalEvaluation }

export async function fetchExperiments(manifestUrl: string): Promise<unknown[]> {
  return Promise.all(['recipes.json', 'evaluation.json'].map(async file => {
    const response = await fetch(new URL(file, manifestUrl));
    if (!response.ok) throw new Error(`Could not load ${file}. Refresh to retry.`);
    return response.json();
  }));
}

export function validateExperiments(data: unknown[], manifest: ModelManifest, tokenizer: ModelTokenizer): ExperimentMetadata {
  const [catalog, evaluation] = data as [{ schemaVersion: number; modelRevision: string; recipes: ExperimentRecipe[] }, RetrievalEvaluation];
  if (catalog?.schemaVersion !== 1 || catalog.modelRevision !== manifest.modelRevision || evaluation?.modelRevision !== manifest.modelRevision ||
      !Array.isArray(catalog.recipes) || catalog.recipes.length !== 3) throw new Error('The prepared experiments do not match this checkpoint.');
  const ids = new Set<string>();
  for (const recipe of catalog.recipes) {
    if (!recipe || typeof recipe.id !== 'string' || ids.has(recipe.id) || typeof recipe.title !== 'string' || typeof recipe.interpretation !== 'string' ||
        recipe.modelRevision !== manifest.modelRevision) throw new Error('Invalid prepared experiment.');
    ids.add(recipe.id);
    validateTokens(recipe.tokenIds, manifest.config);
    validateInterventions([recipe.intervention], manifest.config, recipe.tokenIds.length);
    if (JSON.stringify(tokenize(recipe.text, tokenizer, manifest.config.contextLength)) !== JSON.stringify(recipe.tokenIds) ||
        !Number.isInteger(recipe.predictionPosition) || recipe.predictionPosition < 0 || recipe.predictionPosition >= recipe.tokenIds.length ||
        !Number.isInteger(recipe.targetTokenId) || recipe.targetTokenId < 0 || recipe.targetTokenId >= tokenizer.tokens.length)
      throw new Error('Invalid experiment input or target.');
    if (recipe.intervention.tokens.kind === 'indices' && recipe.intervention.tokens.indices.length !== 1) throw new Error('Prepared experiments require one selected position or all positions.');
    const { metrics } = recipe;
    if (!metrics || [metrics.baselineLogits, metrics.modifiedLogits].some(values => !Array.isArray(values) || values.length !== tokenizer.tokens.length || values.some(value => !Number.isFinite(value))) ||
        [metrics.baselineTargetProbability, metrics.modifiedTargetProbability, metrics.targetLogProbabilityDelta, metrics.jsDivergence, metrics.maxAbsoluteLogitChange].some(value => !Number.isFinite(value)))
      throw new Error('Invalid experiment measurements.');
    if ([metrics.baselineTargetProbability, metrics.modifiedTargetProbability].some(value => value < 0 || value > 1) || metrics.jsDivergence < 0 || metrics.maxAbsoluteLogitChange < 0) throw new Error('Invalid experiment measurement range.');
  }
  if (!Number.isInteger(evaluation.heldOut?.count) || evaluation.heldOut.count <= 0 || !Number.isFinite(evaluation.heldOut.lossNats) ||
      [evaluation.heldOut.accuracy, evaluation.localContext?.accuracy, evaluation.lengthGeneralization?.['2']?.accuracy, evaluation.lengthGeneralization?.['4']?.accuracy]
        .some(value => !Number.isFinite(value) || value < 0 || value > 1)) throw new Error('Invalid checkpoint evaluation.');
  return { recipes: catalog.recipes, evaluation };
}

export function retrievalAnswer(tokens: number[], tokenizer: ModelTokenizer): number | undefined {
  if (tokenizer.kind !== 'symbols') return;
  const symbols = tokens.slice(1).map(id => tokenizer.tokens[id]);
  if (symbols.length < 11 || symbols.slice(-3).join(' ') !== `? ${symbols.at(-2)} =` || (symbols.length - 3) % 4 !== 0) return;
  const bindings = new Map<string, string>();
  for (let i = 0; i < symbols.length - 3; i += 4) {
    if (!/^[A-D]$/u.test(symbols[i]) || !/^[0-3]$/u.test(symbols[i + 2]) || symbols[i + 1] !== '=' || symbols[i + 3] !== ';' || bindings.has(symbols[i])) return;
    bindings.set(symbols[i], symbols[i + 2]);
  }
  const answer = bindings.get(symbols.at(-2)!);
  return answer === undefined ? undefined : tokenizer.tokens.indexOf(answer);
}
