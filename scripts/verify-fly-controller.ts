// Usage: npx tsx scripts/verify-fly-controller.ts [cnn weights .bin ...]   (defaults to public/cnn.bin if it exists)
// Adds a tuned fly circuit row if data/tuning.json exists. Prints the table and writes data/benchmark.json.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createFlyController, stepFlyController, type FlyControllerConfig } from '../src/controllers/flyCircuit';
import { createCnnController, loadCnnWeights, stepCnnController } from '../src/controllers/cnn';
import { BENCH_SEEDS } from '../src/sim/seeds';
import { DT, flyEpisode, type Steer } from './corridor-harness';
import { expertSteer } from './expert';

// Benchmark seeds: never trained or tuned on, and not used for checkpoint selection either.
const SEEDS = Array.from({ length: 20 }, (_, i) => BENCH_SEEDS.from + i);

let escapes = 0;
let effort = 0;
let steps = 0;

function circuit(rmoEnabled: boolean, drive: boolean, tuning: Partial<FlyControllerConfig> = {}): () => Steer {
  return () => {
    const controller = createFlyController({ rmoEnabled, ...tuning });
    return (frame) => {
      const { steering, debug } = stepFlyController(controller, frame, DT);
      if (debug.escaping) escapes++;
      effort += Math.abs(steering);
      steps++;
      return drive ? steering : 0;
    };
  };
}

const expert: () => Steer = () => (_frame, drone, obstacles) => {
  const steering = expertSteer(drone, obstacles);
  effort += Math.abs(steering);
  steps++;
  return steering;
};

const rows: [string, () => Steer, boolean][] = [
  ['no controller', circuit(true, false), true],
  ['fly, RMO on', circuit(true, true), true],
  ['fly, RMO off', circuit(false, true), true],
  ['expert (cheat)', expert, false],
];

if (existsSync('data/tuning.json')) {
  const best = JSON.parse(readFileSync('data/tuning.json', 'utf8')).trials[0].tuning;
  rows.push(['fly, tuned', circuit(true, true, best), true]);
}

const cnnPaths = process.argv.slice(2);
if (cnnPaths.length === 0 && existsSync('public/cnn.bin')) cnnPaths.push('public/cnn.bin');
for (const path of cnnPaths) {
  const weights = loadCnnWeights(readFileSync(path));
  rows.push([
    path,
    () => {
      const controller = createCnnController(weights);
      return (frame) => {
        const steering = stepCnnController(controller, frame, DT);
        effort += Math.abs(steering);
        steps++;
        return steering;
      };
    },
    false,
  ]);
}

console.log(`Flying ${SEEDS.length} benchmark corridors (seeds ${SEEDS[0]}-${SEEDS[SEEDS.length - 1]}).\n`);
console.log(`${'condition'.padEnd(28)}collisions   escape-frames   avg|steering|`);

const results = [];
for (const [label, makeSteer, hasEscapes] of rows) {
  escapes = effort = steps = 0;
  let collisions = 0;
  for (const seed of SEEDS) collisions += flyEpisode(seed, makeSteer()).collisions;
  const avgSteering = effort / steps;
  results.push({ label, collisions, escapes: hasEscapes ? escapes : null, avgSteering });
  console.log(
    `${label.padEnd(28)}${String(collisions).padStart(10)}   ${(hasEscapes ? String(escapes) : '-').padStart(13)}   ${avgSteering.toFixed(3).padStart(13)}`,
  );
}

mkdirSync('data', { recursive: true });
writeFileSync('data/benchmark.json', JSON.stringify(results, null, 2));
