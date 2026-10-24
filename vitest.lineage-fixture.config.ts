import { defineConfig } from 'vitest/config';
export default defineConfig({test:{include:['scripts/export_lineage_workspace.export.ts'],maxWorkers:1}});
