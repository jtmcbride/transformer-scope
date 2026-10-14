import { requireInput, type ModelTokenizer } from './contracts';
import { BpeTokenizer } from './bpe';

const encoders = new WeakMap<object, BpeTokenizer>();
function bpe(tokenizer: ModelTokenizer): BpeTokenizer {
  requireInput(tokenizer.kind === 'byte-bpe', 'Use the text tokenizer for this checkpoint.');
  let encoder = encoders.get(tokenizer);
  if (!encoder) { encoder = new BpeTokenizer(tokenizer); encoders.set(tokenizer, encoder); }
  return encoder;
}
/** Both checkpoint kinds use BOS at position zero; text preserves every input byte. */
export function tokenize(text: string, tokenizer: ModelTokenizer, contextLength: number): number[] {
  if (tokenizer.kind === 'byte-bpe') {
    requireInput(text.length <= 65_536, 'Text input is too large. Use a shorter passage.');
    const ids = bpe(tokenizer).encode(text);
    requireInput(ids.length <= contextLength, `This passage uses ${ids.length} tokens. Use at most ${contextLength - 1} text tokens, plus the beginning token; shorten the passage.`);
    return ids;
  }
  const symbols = text.trim() ? text.trim().split(/\s+/u) : [];
  const vocabulary = new Map(tokenizer.tokens.map((token, id) => [token, id]));
  const tokens = [tokenizer.bosTokenId, ...symbols.map(symbol => {
    const id = vocabulary.get(symbol);
    requireInput(id !== undefined && id !== tokenizer.bosTokenId, `Unknown symbol "${symbol}". Use the displayed vocabulary.`);
    return id;
  })];
  requireInput(tokens.length <= contextLength, `Use at most ${contextLength - 1} symbols; the beginning-of-sequence token uses one position.`);
  return tokens;
}
export function textForTokens(ids: readonly number[], tokenizer: ModelTokenizer): string {
  return tokenizer.kind === 'byte-bpe' ? bpe(tokenizer).decode(ids) : ids.slice(1).map(id => tokenizer.tokens[id]).join(' ');
}
/** Whitespace markers are for labels only, never for encoding or replay. */
export function tokenLabel(tokenizer: ModelTokenizer, id: number): string {
  const text = tokenizer.tokens[id];
  return tokenizer.kind === 'byte-bpe' ? text.replaceAll(' ', '␠').replaceAll('\n', '↵').replaceAll('\r', '␍').replaceAll('\t', '⇥') : text;
}
