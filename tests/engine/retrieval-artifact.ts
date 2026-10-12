import { readFileSync } from 'node:fs';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import type { ModelArtifact, TensorRecord } from '../../src/engine/contracts';

/** A shipped checkpoint, loaded from public/ for Node engine tests and analyses. */
export function loadShippedArtifact(name: 'retrieval' | 'verification') {
  const directory = new URL(`../../public/models/${name}/`, import.meta.url);
  const read = (file: string) => JSON.parse(readFileSync(new URL(file, directory), 'utf8'));
  const manifest = validateManifest(read('manifest.json'));
  const tokenizer = validateTokenizer(read('tokenizer.json'), manifest.config.vocabularySize);
  const payload = readFileSync(new URL('weights.bin', directory));
  const weights: Record<string, TensorRecord> = {};
  for (const entry of manifest.tensors) weights[entry.name] = { shape: entry.shape,
    values: Array.from({ length: entry.byteLength / 4 }, (_, i) => payload.readFloatLE(entry.byteOffset + i * 4)) };
  return { read, manifest, tokenizer, artifact: { manifest, tokenizer, weights } as ModelArtifact };
}

const retrieval = loadShippedArtifact('retrieval');
export const json = retrieval.read;
export const { manifest, tokenizer, artifact } = retrieval;
