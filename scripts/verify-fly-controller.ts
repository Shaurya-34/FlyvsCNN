// Usage: npx tsx scripts/verify-fly-controller.ts [cnn weights .bin ...] [--three] [--eval]  (defaults to public/cnn.bin)
// Adds a tuned fly circuit row if data/tuning.json exists. Prints the table and writes data/benchmark.json,
// or with --eval runs the held-out seeds once and writes data/evaluation.json (Phase 7).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import type { ContestantSpec } from '../src/bench/tasks';
import { BENCH_SEEDS, EVAL_SEEDS } from '../src/sim/seeds';
import { args, openRunner, useThree } from './runner';

// Benchmark seeds: never trained or tuned on, and not used for checkpoint selection either.
// Eval seeds: touched only here, and only once the circuit and the CNNs are final.
const evaluating = args.includes('--eval');
const SEEDS = evaluating
  ? Array.from({ length: 100 }, (_, i) => EVAL_SEEDS.from + i)
  : Array.from({ length: 20 }, (_, i) => BENCH_SEEDS.from + i);

const rows: [string, ContestantSpec][] = [
  ['no controller', { kind: 'fly', drive: false, tuning: {} }],
  ['fly, RMO on', { kind: 'fly', drive: true, tuning: {} }],
  ['fly, RMO off', { kind: 'fly', drive: true, tuning: { rmoEnabled: false } }],
  ['fly, no inhibition', { kind: 'fly', drive: true, tuning: { inhibitionEnabled: false } }],
  ['expert (cheat)', { kind: 'expert' }],
];

if (existsSync('data/tuning.json')) {
  const best = JSON.parse(readFileSync('data/tuning.json', 'utf8')).trials[0].tuning;
  rows.push(['fly, tuned', { kind: 'fly', drive: true, tuning: best }]);
}

const cnnPaths = args.filter((a) => !a.startsWith('--'));
if (cnnPaths.length === 0 && existsSync('public/cnn.bin')) cnnPaths.push('public/cnn.bin');
for (const path of cnnPaths) rows.push([path, { kind: 'cnn', path }]);

console.log(
  `Flying ${SEEDS.length} ${evaluating ? 'held-out' : 'benchmark'} corridors (seeds ${SEEDS[0]}-${SEEDS[SEEDS.length - 1]}) ` +
    `on the ${useThree ? 'Three.js' : 'harness'} renderer.\n`,
);
console.log(`${'condition'.padEnd(28)}collisions   escape-frames   avg|steering|`);

const runner = await openRunner();
const results = [];
for (const [label, spec] of rows) {
  let collisions = 0;
  let escapes = 0;
  let effort = 0;
  let steps = 0;
  for (const seed of SEEDS) {
    const r = await runner.benchmark(seed, spec);
    collisions += r.collisions;
    escapes += r.escapes;
    effort += r.effort;
    steps += r.steps;
  }
  const hasEscapes = spec.kind === 'fly';
  const avgSteering = effort / steps;
  results.push({ label, collisions, escapes: hasEscapes ? escapes : null, avgSteering });
  console.log(
    `${label.padEnd(28)}${String(collisions).padStart(10)}   ${(hasEscapes ? String(escapes) : '-').padStart(13)}   ${avgSteering.toFixed(3).padStart(13)}`,
  );
}
await runner.close();

mkdirSync('data', { recursive: true });
writeFileSync(evaluating ? 'data/evaluation.json' : 'data/benchmark.json', JSON.stringify(results, null, 2));
