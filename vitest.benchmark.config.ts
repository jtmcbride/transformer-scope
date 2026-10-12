import { defineConfig } from 'vitest/config';

export default defineConfig({ test: { include: ['tests/benchmarks/*.benchmark.ts'], maxWorkers: 1 } });
