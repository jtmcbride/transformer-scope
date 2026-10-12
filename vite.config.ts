import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/transformer-scope/',
  worker: { format: 'es' },
  // The validation harness and study session are separate opt-in pages; the application entry imports neither.
  build: { rollupOptions: { input: { main: 'index.html', validation: 'validation.html', study: 'study.html' } } },
});
