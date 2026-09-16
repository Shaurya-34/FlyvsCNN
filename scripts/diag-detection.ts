// Does the fly circuit see obstacles at a useful distance, or only when they fill the view?
// Flies corridors, then compares its reported object size and escape timing against the real geometry.
// Usage: npx tsx scripts/diag-detection.ts [episodes=10] [--tuned] [--three]
import { readFileSync } from 'node:fs';
import { DRONE_RADIUS, obstacleRadius } from '../src/sim/collisions';
import { generateCorridor } from '../src/sim/corridor';
import { CORRIDOR } from '../src/sim/episode';
import { BENCH_SEEDS } from '../src/sim/seeds';
import { TRACE_COLUMNS } from '../src/bench/tasks';
import { args, openRunner } from './runner';

const LOOK_AHEAD = 12; // metres of corridor ahead counted as "in front of the drone"
const DODGE_DISTANCE = 2; // closer than this, at 4 m/s, there is no time left to get out of the way
const HALF_FOV_DEG = 45;

const episodes = Number(args.find((a) => !a.startsWith('--')) ?? 10);
const tuned = args.includes('--tuned');
const tuning = tuned ? JSON.parse(readFileSync('data/tuning.json', 'utf8')).trials[0].tuning : {};

const deg = (rad: number) => (rad * 180) / Math.PI;
const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : NaN);

const runner = await openRunner();
const sizeRatios: number[] = [];
const detectionDistances: number[] = [];
let threats = 0;
let lateOrMissed = 0;
let escapeSteps = 0;
let escapesAtNothing = 0; // nothing at all in view
let escapesAtBystander = 0; // something in view, but the drone would miss it anyway

for (let seed = BENCH_SEEDS.from; seed < BENCH_SEEDS.from + episodes; seed++) {
  const obstacles = generateCorridor({ seed, ...CORRIDOR });
  const trace = await runner.trace(seed, tuning);
  const firstEscapeAt = new Map<number, number>(); // obstacle index -> distance when the circuit first escaped at it

  for (let row = 0; row * TRACE_COLUMNS < trace.length; row++) {
    const [droneX, droneZ, sizeDeg, , escaping] = trace.subarray(row * TRACE_COLUMNS, (row + 1) * TRACE_COLUMNS);

    // The nearest obstacle in view, and whether the drone is on course to hit it.
    let nearest = -1;
    let nearestDz = Infinity;
    for (let i = 0; i < obstacles.length; i++) {
      const dz = obstacles[i].z - droneZ;
      if (dz <= 0 || dz > LOOK_AHEAD || dz >= nearestDz) continue;
      if (Math.abs(deg(Math.atan2(obstacles[i].x - droneX, dz))) > HALF_FOV_DEG) continue;
      nearest = i;
      nearestDz = dz;
    }

    if (escaping) escapeSteps++;
    if (nearest < 0) {
      if (escaping) escapesAtNothing++;
      continue;
    }

    const o = obstacles[nearest];
    const radius = obstacleRadius(o);
    const onCourse = Math.abs(o.x - droneX) < radius + DRONE_RADIUS;
    if (escaping && !onCourse) escapesAtBystander++;

    // What the circuit should be reporting: the obstacle's angular size as an equal-area circle.
    const widthDeg = 2 * deg(Math.atan(radius / nearestDz));
    const heightDeg = deg(Math.atan((o.height - 0.6) / nearestDz) + Math.atan(0.6 / nearestDz));
    const trueSizeDeg = 2 * Math.sqrt((widthDeg * heightDeg) / Math.PI);
    if (nearestDz < 6) sizeRatios.push(sizeDeg / trueSizeDeg);

    if (onCourse) {
      if (!firstEscapeAt.has(nearest)) firstEscapeAt.set(nearest, escaping ? nearestDz : -1);
      else if (firstEscapeAt.get(nearest) === -1 && escaping) firstEscapeAt.set(nearest, nearestDz);
    }
  }

  for (const distance of firstEscapeAt.values()) {
    threats++;
    if (distance < DODGE_DISTANCE) lateOrMissed++;
    else detectionDistances.push(distance);
  }
}

await runner.close();

console.log(`${episodes} corridors, fly circuit ${tuned ? '(tuned)' : '(default)'}`);
console.log(`reported size / true size, within 6 m:  median ${median(sizeRatios).toFixed(2)}`);
console.log(`obstacles on a collision course:        ${threats}`);
console.log(`  seen with room to dodge (>${DODGE_DISTANCE} m):     ${threats - lateOrMissed} (median ${median(detectionDistances).toFixed(1)} m)`);
console.log(`  seen too late or never:               ${lateOrMissed} (${((100 * lateOrMissed) / threats).toFixed(0)}%)`);
const pct = (n: number) => `${((100 * n) / escapeSteps).toFixed(0)}%`;
console.log(`escape steps:                           ${escapeSteps}`);
console.log(`  nothing in view at all:               ${pct(escapesAtNothing)}`);
console.log(`  obstacle in view but not on course:   ${pct(escapesAtBystander)}`);
console.log(`  on a collision course:                ${pct(escapeSteps - escapesAtNothing - escapesAtBystander)}`);
