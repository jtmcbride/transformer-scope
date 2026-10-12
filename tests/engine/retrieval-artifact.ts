import { readFileSync } from 'node:fs';
import { validateManifest, validateTokenizer } from '../../src/engine/weights';
import type { ModelArtifact, TensorRecord } from '../../src/engine/contracts';

/** The shipped trained checkpoint, loaded from public/ for Node engine tests and analyses. */
const directory = new URL('../../public/models/retrieval/', import.meta.url);
export const json = (file: string) => JSON.parse(readFileSync(new URL(file, directory), 'utf8'));
export const manifest = validateManifest(json('manifest.json'));
export const tokenizer = validateTokenizer(json('tokenizer.json'), manifest.config.vocabularySize);
const payload = readFileSync(new URL('weights.bin', directory));
const weights: Record<string, TensorRecord> = {};
for (const entry of manifest.tensors) weights[entry.name] = { shape: entry.shape,
  values: Array.from({ length: entry.byteLength / 4 }, (_, i) => payload.readFloatLE(entry.byteOffset + i * 4)) };
export const artifact: ModelArtifact = { manifest, tokenizer, weights };
