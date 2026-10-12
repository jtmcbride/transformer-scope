import { defineConfig } from '@playwright/test';
import base from './playwright.config';

export default defineConfig({ ...base, testDir: 'tests/benchmarks', testMatch: 'browser.benchmark.ts', workers: 1, fullyParallel: false,
  reporter: [['list'], ['json', { outputFile: '.cache/browser-benchmark-results.json' }]],
  projects: base.projects!.filter(project => project.name !== 'firefox'),
});
