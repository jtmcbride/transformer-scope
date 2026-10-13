import { requireInput } from './contracts';

/** Byte-level BPE, format 2. Specification: scripts/text_tokenizer.py (the independent reference). */
export interface BpeDocument {
  schemaVersion: 2; kind: 'byte-bpe'; pattern: string; bosTokenId: 0; eosTokenId: 1; padTokenId: 2; byteOffset: 3;
  merges: [number, number][]; tokens: string[];
}
export const BPE_PATTERN = ' ?[A-Za-z]+| ?[0-9]| ?[^A-Za-z0-9 \\t\\n\\r]+|[ \\t\\n\\r]+';

export function validateBpe(value: unknown, vocabularySize: number): BpeDocument {
  const document = value as BpeDocument;
  requireInput(document?.schemaVersion === 2 && document.kind === 'byte-bpe' && document.pattern === BPE_PATTERN, 'Unsupported text tokenizer.');
  requireInput(document.bosTokenId === 0 && document.eosTokenId === 1 && document.padTokenId === 2 && document.byteOffset === 3, 'Unexpected special token layout.');
  requireInput(Array.isArray(document.merges) && 3 + 256 + document.merges.length === vocabularySize &&
    Array.isArray(document.tokens) && document.tokens.length === vocabularySize, 'Tokenizer size does not match the model vocabulary.');
  document.merges.forEach(([a, b], rank) => requireInput(Number.isInteger(a) && Number.isInteger(b) && a >= 3 && b >= 3 && a < 259 + rank && b < 259 + rank,
    'Merges may only combine bytes or earlier merges.'));
  return document;
}

export class BpeTokenizer {
  private readonly ranks = new Map<number, number>();
  private readonly bytes: Uint8Array[] = [];
  private readonly cache = new Map<string, number[]>();
  private readonly splitter: RegExp;
  readonly tokens: readonly string[];

  constructor(readonly document: BpeDocument) {
    this.splitter = new RegExp(document.pattern, 'gu');
    this.tokens = document.tokens;
    for (let i = 0; i < document.byteOffset; i++) this.bytes.push(new Uint8Array());
    for (let b = 0; b < 256; b++) this.bytes.push(Uint8Array.of(b));
    document.merges.forEach(([a, b], rank) => {
      this.ranks.set(a * 65536 + b, rank);
      const joined = new Uint8Array(this.bytes[a].length + this.bytes[b].length);
      joined.set(this.bytes[a]); joined.set(this.bytes[b], this.bytes[a].length);
      this.bytes.push(joined);
    });
  }

  private encodePiece(piece: string): number[] {
    const cached = this.cache.get(piece);
    if (cached) return cached;
    let sequence = Array.from(new TextEncoder().encode(piece), byte => byte + this.document.byteOffset);
    while (sequence.length > 1) {
      let best = Infinity, left = -1, right = -1;
      for (let i = 0; i + 1 < sequence.length; i++) {
        const rank = this.ranks.get(sequence[i] * 65536 + sequence[i + 1]);
        if (rank !== undefined && rank < best) { best = rank; left = sequence[i]; right = sequence[i + 1]; }
      }
      if (best === Infinity) break;
      // Merge every non-overlapping occurrence of the lowest-ranked pair, left to right.
      const merged: number[] = [], id = this.document.byteOffset + 256 + best;
      for (let i = 0; i < sequence.length; i++) {
        if (i + 1 < sequence.length && sequence[i] === left && sequence[i + 1] === right) { merged.push(id); i++; }
        else merged.push(sequence[i]);
      }
      sequence = merged;
    }
    this.cache.set(piece, sequence);
    return sequence;
  }

  encode(text: string, options: { bos?: boolean; eos?: boolean } = {}): number[] {
    // UTF-8 cannot represent an unpaired surrogate; the Python reference rejects it too.
    requireInput(!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(text), 'Text contains an unpaired surrogate.');
    const ids: number[] = options.bos === false ? [] : [this.document.bosTokenId];
    for (const match of text.matchAll(this.splitter)) ids.push(...this.encodePiece(match[0]));
    if (options.eos) ids.push(this.document.eosTokenId);
    return ids;
  }

  decode(ids: readonly number[]): string {
    const parts = ids.filter(id => id >= this.document.byteOffset).map(id => this.bytes[id]);
    const joined = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
    let offset = 0;
    for (const part of parts) { joined.set(part, offset); offset += part.length; }
    return new TextDecoder('utf-8', { fatal: true }).decode(joined);
  }
}
