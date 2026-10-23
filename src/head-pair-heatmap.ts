import type { ModelConfig } from './engine/contracts';
import type { HeadPair, HeadPairScanRow } from './story-head-pair-scan';

export const headPairKey = (edits: HeadPair) => edits.map(edit => `${edit.layer}:${edit.head}:${edit.factor}`).join('|');

/** Exact assignments, never an average or maximum across strengths. Missing is not zero. */
export function headPairHeatmap(rows: HeadPairScanRow[], config: ModelConfig, rowFactor: number, columnFactor: number) {
  const count = config.layers * config.heads, indexed = new Map(rows.map(row => [headPairKey(row.edits), row]));
  const seen = new Set<HeadPairScanRow>();
  const cells = Array.from({ length: count }, (_, r) => Array.from({ length: count }, (_, c) => {
    if (r === c) return undefined;
    const a = `${Math.floor(r/config.heads)}:${r%config.heads}:${rowFactor}`;
    const b = `${Math.floor(c/config.heads)}:${c%config.heads}:${columnFactor}`;
    const result = indexed.get(r < c ? `${a}|${b}` : `${b}|${a}`);
    if (result) seen.add(result);
    return result;
  }));
  // One symmetric scale across the whole scan, so changing strength views retains color meaning.
  const limit = rows.reduce((max, row) => Math.max(max, Math.abs(row.interaction)), 0);
  return { cells, limit, measured: seen.size, total: count*(count-1)/(rowFactor === columnFactor ? 2 : 1) };
}

function interactionRGB(value: number, limit: number): number[] {
  const neutral = [246,247,250], end = value < 0 ? [36,91,151] : [176,75,24];
  const strength = limit > 0 ? Math.min(1,Math.abs(value)/limit) : 0;
  return neutral.map((start,i) => Math.round(start+(end[i]-start)*strength));
}
export const interactionFill = (value: number, limit: number) => `rgb(${interactionRGB(value,limit).join(', ')})`;
/** Keep the sign readable at both saturated endpoints and intermediate colors. */
export function interactionInk(value: number, limit: number): string {
  const linear = interactionRGB(value,limit).map(v => v/255).map(v => v <= .04045 ? v/12.92 : ((v+.055)/1.055)**2.4);
  const luminance = .2126*linear[0]+.7152*linear[1]+.0722*linear[2];
  return 1.05/(luminance+.05) >= 4.5 ? '#fff' : '#000';
}
