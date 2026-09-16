import { defineConfig } from 'vite';

// Default build: the standalone page in dist/.
// `npm run build:embed` instead emits dist/flyvscnn.js, one self-mounting bundle (Three.js included) for the blog.
export default defineConfig(({ mode }) => ({
  base: './',
  build:
    mode === 'embed'
      ? {
          lib: { entry: 'src/widget/main.ts', formats: ['es'], fileName: () => 'flyvscnn.js' },
          emptyOutDir: false,
        }
      : {},
}));
