import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/transformer-scope/',
  worker: { format: 'es' },
  // The validation harness is a separate opt-in page; the application entry does not import it.
  build: { rollupOptions: { input: { main: 'index.html', validation: 'validation.html' } } },
});
