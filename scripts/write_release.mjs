import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const directory = 'dist';
function entries(path) {
  return readdirSync(path, { withFileTypes: true }).flatMap(entry => {
    const file = join(path, entry.name);
    if (entry.isDirectory()) return entries(file);
    if (!entry.isFile() || entry.name === 'release.json') return [];
    const bytes = readFileSync(file);
    return [{ path: relative(directory, file).replaceAll('\\', '/'), byteLength: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex') }];
  });
}
const commit = process.env.GITHUB_SHA ?? execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const dirty = execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim().length > 0;
const models = Object.fromEntries(['verification', 'retrieval', 'tinystories'].map(name => {
  const manifest = JSON.parse(readFileSync(join(directory, 'models', name, 'manifest.json'), 'utf8'));
  return [name, { modelId: manifest.modelId, modelRevision: manifest.modelRevision }];
}));
const release = { format: 'transformer-microscope-release', schemaVersion: 1, commit, sourceTreeDirty: dirty,
  builtAt: new Date().toISOString(), basePath: '/transformer-scope/', models,
  assets: entries(directory).sort((a, b) => a.path.localeCompare(b.path)) };
writeFileSync(join(directory, 'release.json'), JSON.stringify(release, null, 2) + '\n');
console.log(`Stamped ${release.assets.length} deployment assets for ${commit.slice(0, 7)}${dirty ? ' (working-tree changes)' : ''}.`);
