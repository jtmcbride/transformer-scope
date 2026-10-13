import { defineConfig } from 'vitest/config';

export default defineConfig({ test: { include: ['tests/text/*.parity.ts'], maxWorkers: 1 } });
