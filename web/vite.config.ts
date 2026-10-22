import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  plugins: [react(), viteStaticCopy({ targets: ['cmaps', 'standard_fonts', 'wasm', 'iccs'].map(directory => ({
    src: fileURLToPath(new URL(`./node_modules/pdfjs-dist/${directory}`, import.meta.url)), dest: 'pdfjs',
  })) })],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});
