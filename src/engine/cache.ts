import type { LayerBaseline } from './model';
import { requireInput } from './contracts';

interface Entry { identity: string; layers: LayerBaseline[]; bytes: number; pins: number; retired: boolean }
export interface CacheLease { layers: readonly LayerBaseline[]; release(): void }
export interface CacheStats { logicalBytes: number; budgetBytes: number; baselines: number; pinnedEntries: number; retiredEntries: number }

/** Owns admitted and rejected candidates; leases delay disposal but never exceed the budget. */
export class BaselineCache {
  private entries = new Map<string, Entry>();
  private retired = new Set<Entry>();
  private bytes = 0;

  constructor(readonly budgetBytes = 32 * 1024 * 1024) {
    requireInput(Number.isSafeInteger(budgetBytes) && budgetBytes >= 0, 'Cache budget must be a nonnegative integer number of bytes.');
  }

  put(runId: string, identity: string, layers: LayerBaseline[]): boolean {
    this.delete(runId);
    const entry: Entry = { identity, layers, bytes: layers.reduce((total, layer) =>
      total + (layer.rawHeadOutputs.size + layer.attentionResidual.size) * 4, 0), pins: 0, retired: false };
    if (entry.bytes > this.budgetBytes) { this.disposeLayers(layers); return false; }
    for (const [id, candidate] of this.entries) {
      if (this.bytes + entry.bytes <= this.budgetBytes) break;
      if (candidate.pins === 0) this.delete(id);
    }
    if (this.bytes + entry.bytes > this.budgetBytes) { this.disposeLayers(layers); return false; }
    this.entries.set(runId, entry); this.bytes += entry.bytes;
    return true;
  }

  acquire(runId: string, identity: string): CacheLease | undefined {
    const entry = this.entries.get(runId);
    if (!entry) return undefined;
    if (entry.identity !== identity) { this.delete(runId); return undefined; }
    this.entries.delete(runId); this.entries.set(runId, entry);
    entry.pins++;
    let released = false;
    return { layers: entry.layers, release: () => {
      if (released) return;
      released = true; entry.pins--;
      if (entry.retired && entry.pins === 0) this.disposeEntry(entry);
    } };
  }

  delete(runId: string): void {
    const entry = this.entries.get(runId);
    if (!entry) return;
    this.entries.delete(runId);
    entry.retired = true;
    if (entry.pins) this.retired.add(entry);
    else this.disposeEntry(entry);
  }

  clear(): void { for (const id of this.entries.keys()) this.delete(id); }

  stats(): CacheStats {
    return { logicalBytes: this.bytes, budgetBytes: this.budgetBytes, baselines: this.entries.size,
      pinnedEntries: [...this.entries.values(), ...this.retired].filter(entry => entry.pins > 0).length, retiredEntries: this.retired.size };
  }

  private disposeLayers(layers: readonly LayerBaseline[]): void {
    new Set(layers.flatMap(layer => [layer.rawHeadOutputs, layer.attentionResidual])).forEach(tensor => tensor.dispose());
  }

  private disposeEntry(entry: Entry): void {
    this.disposeLayers(entry.layers); this.bytes -= entry.bytes; this.retired.delete(entry);
  }
}
