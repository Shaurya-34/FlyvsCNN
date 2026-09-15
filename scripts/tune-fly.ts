// Usage: npx tsx scripts/tune-fly.ts [trials=100]
// Random search over the fly circuit's engineering constants; the published biology constants stay fixed.
// Scores on training corridors only (the same 20 the 20-episode CNN learned from). Writes data/tuning.json.
import { mkdirSync, writeFileSync } from 'node:fs';
import { DEFAULT_TUNING, createFlyController, stepFlyController, type FlyTuning } from '../src/controllers/flyCircuit';
import { mulberry32 } from '../src/sim/corridor';
import { TRAIN_SEEDS } from '../src/sim/seeds';
import { DT, flyEpisode } from './corridor-harness';

const TRIALS = Number(process.argv[2] ?? 100);
const TUNE_SEEDS = Array.from({ length: 20 }, (_, i) => TRAIN_SEEDS.from + i);
// Collision-equivalent cost of steering at full deflection the whole time. Scoring collisions alone let the search win
// by swerving constantly (94% of frames in escape), so effort has to cost something.
const STEERING_PENALTY = 50;

// Each constant is sampled log-uniformly in [min, max].
const RANGES: Record<keyof FlyTuning, [number, number]> = {
  escapeThreshold: [0.3, 3],
  steeringGain: [0.05, 3],
  evasiveDurationS: [0.05, 1],
  hysteresis: [0.5, 0.95],
  excessMargin: [0.5, 3],
  componentRatio: [0.1, 0.7],
  gradientFloorSq: [0.001, 0.02],
};

function score(tuning: FlyTuning) {
  let collisions = 0;
  let effort = 0;
  let steps = 0;
  for (const seed of TUNE_SEEDS) {
    const controller = createFlyController(tuning);
    collisions += flyEpisode(seed, (frame) => {
      const { steering } = stepFlyController(controller, frame, DT);
      effort += Math.abs(steering);
      steps++;
      return steering;
    }).collisions;
  }
  const avgSteering = effort / steps;
  return { collisions, avgSteering, objective: collisions + STEERING_PENALTY * avgSteering };
}

const rng = mulberry32(12345);
const sample = () =>
  Object.fromEntries(
    Object.entries(RANGES).map(([key, [lo, hi]]) => [key, Math.exp(Math.log(lo) + rng() * (Math.log(hi) - Math.log(lo)))]),
  ) as FlyTuning;

const fmt = (r: ReturnType<typeof score>) =>
  `objective ${r.objective.toFixed(1)}  (collisions ${r.collisions}, avg|steering| ${r.avgSteering.toFixed(3)})`;

const trials = [{ tuning: DEFAULT_TUNING, ...score(DEFAULT_TUNING) }];
console.log(`default: ${fmt(trials[0])} on ${TUNE_SEEDS.length} training corridors`);

for (let i = 1; i <= TRIALS; i++) {
  const best = Math.min(...trials.map((t) => t.objective));
  const tuning = sample();
  const result = score(tuning);
  trials.push({ tuning, ...result });
  console.log(`trial ${i}: ${fmt(result)}${result.objective < best ? '  best' : ''}`);
}

trials.sort((a, b) => a.objective - b.objective);
mkdirSync('data', { recursive: true });
writeFileSync(
  'data/tuning.json',
  JSON.stringify(
    { tuneSeeds: TUNE_SEEDS, steeringPenalty: STEERING_PENALTY, corridorRunsFlown: trials.length * TUNE_SEEDS.length, trials },
    null,
    2,
  ),
);
console.log(`best: ${fmt(trials[0])}`, trials[0].tuning);
