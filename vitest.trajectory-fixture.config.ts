import { defineConfig } from 'vitest/config';
export default defineConfig({test:{include:['scripts/export_trajectory_workspace.export.ts'],maxWorkers:1}});
