import type { ResidualHook, ModelTokenizer } from './engine/contracts';
import { retrievalAnswer } from './experiments';

/** A patch point is serialized as "layer:stage" for a select control. */
export function parseHook(key: string): { layer: number; stage: ResidualHook } {
  const [layer, stage] = key.split(':');
  return { layer: Number(layer), stage: stage as ResidualHook };
}
export function hookLabel(key: string): string {
  const { layer, stage } = parseHook(key);
  return stage === 'block.input' ? `Layer ${layer + 1} input` : `Layer ${layer + 1} after attention`;
}
export function hookKeys(layers: number): string[] {
  return Array.from({ length: layers }, (_, layer) => [`${layer}:block.input`, `${layer}:attention.residual`]).flat();
}

/** For a well-formed retrieval input, the same bindings queried for another key; otherwise undefined. */
export function swappedQuery(tokens: readonly number[], tokenizer: ModelTokenizer): string | undefined {
  if (tokenizer.kind !== 'symbols') return;
  if (retrievalAnswer([...tokens], tokenizer) === undefined) return;
  const symbols = tokens.slice(1).map(id => tokenizer.tokens[id]);
  const query = symbols.at(-2)!;
  const keys = symbols.filter((_, i) => i < symbols.length - 3 && i % 4 === 0);
  const other = keys.find(key => key !== query && keys.indexOf(key) < keys.length - 1);
  return other ? [...symbols.slice(0, -2), other, '='].join(' ') : undefined;
}
