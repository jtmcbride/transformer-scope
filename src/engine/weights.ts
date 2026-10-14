import { validateBpe } from './bpe';
import { EngineError, type ModelArtifact, type ModelConfig, type ModelManifest, type SymbolTokenizer, type ModelTokenizer, type TensorRecord } from './contracts';

export function expectedShapes(c: ModelConfig): Record<string, number[]> {
  const shapes: Record<string, number[]> = {
    tokenEmbedding: [c.vocabularySize, c.modelDim], positionEmbedding: [c.contextLength, c.modelDim],
    'finalNorm.scale': [c.modelDim], 'finalNorm.bias': [c.modelDim],
  };
  for (let layer = 0; layer < c.layers; layer++) {
    const p = `blocks.${layer}.`;
    for (const norm of ['ln1', 'ln2']) {
      shapes[p + norm + '.scale'] = [c.modelDim]; shapes[p + norm + '.bias'] = [c.modelDim];
    }
    for (const projection of ['q', 'k', 'v', 'output']) {
      shapes[p + projection + '.weight'] = [c.modelDim, c.modelDim]; shapes[p + projection + '.bias'] = [c.modelDim];
    }
    shapes[p + 'mlpIn.weight'] = [c.modelDim, c.mlpDim]; shapes[p + 'mlpIn.bias'] = [c.mlpDim];
    shapes[p + 'mlpOut.weight'] = [c.mlpDim, c.modelDim]; shapes[p + 'mlpOut.bias'] = [c.modelDim];
  }
  return shapes;
}

function compatible(condition: unknown, message: string): asserts condition {
  if (!condition) throw new EngineError('INCOMPATIBLE_MODEL', message);
}
export function validateManifest(value: unknown): ModelManifest {
  compatible(value && typeof value === 'object', 'Missing model manifest.');
  const m = value as ModelManifest;
  compatible(m.schemaVersion === 1 && m.architectureVersion === 'preln-absolute-tied-v1', 'Unsupported model version.');
  compatible(m.dtype === 'float32-le' && m.matrixOrientation === 'input-output', 'Unsupported weight encoding.');
  compatible(m.config && typeof m.config === 'object', 'Missing architecture configuration.');
  const c = m.config;
  for (const key of ['layers', 'heads', 'modelDim', 'mlpDim', 'vocabularySize', 'contextLength'] as const)
    compatible(Number.isInteger(c[key]) && c[key] > 0, `Invalid ${key}.`);
  compatible(c.layers <= 16 && c.modelDim <= 512 && c.mlpDim <= 2048 && c.vocabularySize <= 8192 && c.contextLength <= 512,
    'This model exceeds the initial engine resource limits.');
  compatible(c.modelDim % c.heads === 0 && c.layerNormEpsilon === 1e-5, 'Incompatible head dimensions or normalization.');
  compatible(m.conventions && m.conventions.gelu === 'tanh' && m.conventions.layerNormVariance === 'population' &&
    m.conventions.position === 'learned-absolute' && m.conventions.tiedEmbeddings === true &&
    m.conventions.dropout === 0 && m.conventions.maskValue === -1e9, 'Unsupported numerical conventions.');
  compatible(typeof m.modelId === 'string' && m.modelId.length > 0 && /^[a-f0-9]{64}$/.test(m.modelRevision), 'Missing model identity.');
  compatible(/^[a-f0-9]{64}$/.test(m.weightsSha256) && /^[a-f0-9]{64}$/.test(m.tokenizerSha256), 'Missing artifact checksums.');
  compatible(typeof m.weightsFile === 'string' && /^[a-zA-Z0-9_.-]+$/.test(m.weightsFile) &&
    typeof m.tokenizerFile === 'string' && /^[a-zA-Z0-9_.-]+$/.test(m.tokenizerFile), 'Artifacts must use local filenames.');
  const expected = expectedShapes(c);
  compatible(Array.isArray(m.tensors) && m.tensors.length === Object.keys(expected).length, 'Unexpected tensor count.');
  const seen = new Set<string>();
  let offset = 0;
  for (const entry of m.tensors) {
    compatible(entry && typeof entry.name === 'string' && !seen.has(entry.name) && Object.hasOwn(expected, entry.name), 'Unknown or duplicate tensor.');
    compatible(Array.isArray(entry.shape) && JSON.stringify(entry.shape) === JSON.stringify(expected[entry.name]), `Invalid shape for ${entry.name}.`);
    compatible(entry.byteOffset === offset && entry.byteLength === entry.shape.reduce((a, b) => a * b, 1) * 4, 'Invalid tensor byte range.');
    offset += entry.byteLength; seen.add(entry.name);
  }
  compatible(offset <= 64 * 1024 * 1024, 'Weights exceed the initial 64 MiB artifact limit.');
  return structuredClone(m);
}

export function validateTokenizer(value: unknown, vocabularySize: number): ModelTokenizer {
  if ((value as { kind?: string })?.kind === 'byte-bpe') {
    try { return structuredClone(validateBpe(value, vocabularySize)); }
    catch (error) { throw new EngineError('INCOMPATIBLE_MODEL', `Invalid text tokenizer: ${error instanceof Error ? error.message : String(error)}`); }
  }
  const t = value as SymbolTokenizer;
  compatible(t && t.schemaVersion === 1 && t.kind === 'symbols' && Array.isArray(t.tokens) &&
    t.tokens.length === vocabularySize && t.tokens.every(token => typeof token === 'string' && token.length > 0 && !/\s/u.test(token)), 'Invalid symbol tokenizer.');
  compatible(new Set(t.tokens).size === t.tokens.length && Number.isInteger(t.bosTokenId) && t.bosTokenId >= 0 && t.bosTokenId < vocabularySize,
    'Invalid tokenizer beginning-of-sequence token or duplicate symbols.');
  return structuredClone(t);
}

export function validateWeightRecords(weights: Record<string, TensorRecord>, config: ModelConfig): void {
  const shapes = expectedShapes(config);
  compatible(weights && Object.keys(weights).length === Object.keys(shapes).length, 'Unexpected weight count.');
  for (const [name, shape] of Object.entries(shapes)) {
    const tensor = weights[name];
    compatible(tensor && JSON.stringify(tensor.shape) === JSON.stringify(shape) && Array.isArray(tensor.values) &&
      tensor.values.length === shape.reduce((a, b) => a * b, 1) && tensor.values.every(Number.isFinite), `Invalid values for ${name}.`);
  }
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
async function checkedFetch(url: URL): Promise<Response> {
  const response = await fetch(url);
  compatible(response.ok, `Model asset failed to load (${response.status}): ${url.pathname}`);
  return response;
}

export async function loadArtifact(manifestUrl: string): Promise<ModelArtifact> {
  const url = new URL(manifestUrl);
  const manifest = validateManifest(await (await checkedFetch(url)).json());
  const [weightResponse, tokenizerResponse] = await Promise.all([
    checkedFetch(new URL(manifest.weightsFile, url)), checkedFetch(new URL(manifest.tokenizerFile, url)),
  ]);
  const [binary, tokenizerBytes] = await Promise.all([weightResponse.arrayBuffer(), tokenizerResponse.arrayBuffer()]);
  const expectedBytes = manifest.tensors.reduce((sum, entry) => sum + entry.byteLength, 0);
  compatible(binary.byteLength === expectedBytes, 'Weight file size does not match the manifest.');
  const [weightHash, tokenizerHash] = await Promise.all([sha256(binary), sha256(tokenizerBytes)]);
  compatible(weightHash === manifest.weightsSha256 && tokenizerHash === manifest.tokenizerSha256, 'Model asset checksum mismatch.');
  const tokenizer = validateTokenizer(JSON.parse(new TextDecoder().decode(tokenizerBytes)), manifest.config.vocabularySize);
  const weights: Record<string, TensorRecord> = {};
  const view = new DataView(binary);
  for (const entry of manifest.tensors) {
    const values = Array.from({ length: entry.byteLength / 4 }, (_, i) => view.getFloat32(entry.byteOffset + i * 4, true));
    weights[entry.name] = { shape: entry.shape, values };
  }
  validateWeightRecords(weights, manifest.config);
  return { manifest, tokenizer, weights };
}
