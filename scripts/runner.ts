// Runs episode tasks on the fast harness renderer, or with --three on the real Three.js renderer in headless Edge
// (a Vite dev server serves bench.html; Playwright drives it).
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { loadCnnWeights, type CnnWeights } from '../src/controllers/cnn';
import { benchmarkEpisode, recordExpertEpisode, traceEpisode, type ContestantSpec } from '../src/bench/tasks';
import type { FlyControllerConfig } from '../src/controllers/flyCircuit';
import { harnessRenderer } from './corridor-harness';

export const useThree = process.argv.includes('--three');
export const args = process.argv.slice(2).filter((a) => a !== '--three');

type Recorded = { frames: Uint8Array; steering: Float32Array; collisions: number };
type Benchmarked = { collisions: number; escapes: number; effort: number; steps: number };

export async function openRunner() {
  if (!useThree) {
    const weights = new Map<string, CnnWeights>();
    const weightsFor = (path: string) => {
      if (!weights.has(path)) weights.set(path, loadCnnWeights(readFileSync(path)));
      return weights.get(path) as CnnWeights;
    };
    return {
      record: async (seed: number): Promise<Recorded> => recordExpertEpisode(seed, harnessRenderer),
      benchmark: async (seed: number, spec: ContestantSpec): Promise<Benchmarked> =>
        benchmarkEpisode(seed, spec, harnessRenderer, weightsFor),
      trace: async (seed: number, tuning: Partial<FlyControllerConfig>): Promise<Float32Array> =>
        traceEpisode(seed, tuning, harnessRenderer),
      close: async () => {},
    };
  }

  const server = await createServer({ logLevel: 'error', server: { port: 5199, hmr: false } });
  await server.listen();
  const browser = await chromium.launch({ channel: 'msedge' });
  const page = await browser.newPage();
  page.on('console', (m) => m.type() === 'error' && console.error('[bench page]', m.text()));
  page.on('pageerror', (e) => console.error('[bench page]', e.message));
  page.on('response', (r) => r.status() >= 400 && console.error('[bench page]', r.status(), r.url()));
  await page.goto(new URL('bench.html', server.resolvedUrls!.local[0]).href);
  await page.waitForFunction(() => 'benchmarkEpisode' in window);

  const bytes = (b64: string) => new Uint8Array(Buffer.from(b64, 'base64'));

  // ponytail: the page sometimes reloads mid-episode (cause not pinned down, happens even with one job running).
  // Episodes are deterministic, so rerunning the interrupted one gives the same result.
  const retryOnReload = async <T>(fn: () => Promise<T>): Promise<T> => {
    try {
      return await fn();
    } catch (e) {
      if (!String(e).includes('Execution context was destroyed')) throw e;
      console.error('[bench page] reloaded mid-episode, rerunning it');
      await page.waitForFunction(() => 'benchmarkEpisode' in window);
      return fn();
    }
  };

  return {
    record: async (seed: number): Promise<Recorded> => {
      const r = await retryOnReload(() => page.evaluate((s) => (window as any).recordExpertEpisode(s), seed));
      return { frames: bytes(r.frames), steering: new Float32Array(bytes(r.steering).buffer), collisions: r.collisions };
    },
    benchmark: (seed: number, spec: ContestantSpec): Promise<Benchmarked> =>
      retryOnReload(() => page.evaluate(([s, c]) => (window as any).benchmarkEpisode(s, c), [seed, spec] as const)),
    trace: async (seed: number, tuning: Partial<FlyControllerConfig>): Promise<Float32Array> => {
      const b64 = await retryOnReload(() =>
        page.evaluate(([s, t]) => (window as any).traceEpisode(s, t), [seed, tuning] as const),
      );
      return new Float32Array(bytes(b64).buffer);
    },
    close: async () => {
      await browser.close();
      await server.close();
    },
  };
}
