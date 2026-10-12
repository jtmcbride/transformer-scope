import { defineConfig } from '@playwright/test';

const baseURL = process.env.HOSTED_BASE_URL ?? 'https://jtmcbride.github.io/transformer-scope/';
const url = new URL(baseURL);
if (!baseURL.endsWith('/') || (url.protocol !== 'https:' && url.hostname !== '127.0.0.1' && url.hostname !== 'localhost'))
  throw new Error('Use an HTTPS deployment URL ending in /, or localhost for testing the harness.');
export default defineConfig({ testDir: 'tests/hosted', timeout: 120_000, workers: 1,
  reporter: [['list'], ['json', { outputFile: 'test-results/hosted-results.json' }]],
  use: { baseURL }, projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
});
