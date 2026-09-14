import { generateCorridor } from '../src/sim/corridor';
import { createDrone, stepDrone } from '../src/sim/drone';
import { checkCollision } from '../src/sim/collisions';
import { createFlyController, stepFlyController } from '../src/controllers/flyCircuit';
import { renderFrame } from './corridor-harness';

const CORRIDOR = { length: 120, width: 8, density: 0.3, sizeVariance: 0.4 };
const DRONE_CONFIG = { speed: 4, lateralDamping: 2 };
const DT = 1 / 60;

function flySeed(seed: number, rmoEnabled: boolean, straightLine = false) {
  const obstacles = generateCorridor({ seed, ...CORRIDOR });
  const drone = createDrone();
  const controller = createFlyController({ hFovDeg: 90, vFovDeg: 90, rmoEnabled, escapeThreshold: 1.0 });

  let collisions = 0;
  let lastHit: number | null = null;
  let escapes = 0;
  let steeringEffort = 0;
  let steps = 0;

  while (drone.z < CORRIDOR.length) {
    const frame = renderFrame(drone.x, drone.z, obstacles);
    const { steering, debug } = stepFlyController(controller, frame, DT);
    if (debug.escaping) escapes++;
    steeringEffort += Math.abs(steering);
    steps++;

    stepDrone(drone, straightLine ? 0 : steering, DT, DRONE_CONFIG);
    // keep it inside the corridor
    drone.x = Math.max(-CORRIDOR.width / 2, Math.min(CORRIDOR.width / 2, drone.x));

    const hit = checkCollision(drone.x, drone.z, obstacles);
    if (hit !== null && hit !== lastHit) collisions++;
    lastHit = hit;
  }

  return { collisions, escapes, avgSteering: steeringEffort / steps };
}

const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8];

console.log('Flying 8 corridors, same seeds for each condition.\n');
console.log('condition        collisions   escape-frames   avg|steering|');

for (const [label, opts] of [
  ['no controller ', { rmo: true, straight: true }],
  ['fly, RMO on   ', { rmo: true, straight: false }],
  ['fly, RMO off  ', { rmo: false, straight: false }],
] as const) {
  let totalCollisions = 0;
  let totalEscapes = 0;
  let totalSteering = 0;
  for (const seed of SEEDS) {
    const r = flySeed(seed, opts.rmo, opts.straight);
    totalCollisions += r.collisions;
    totalEscapes += r.escapes;
    totalSteering += r.avgSteering;
  }
  console.log(
    `${label}   ${String(totalCollisions).padStart(6)}   ${String(totalEscapes).padStart(13)}   ${(totalSteering / SEEDS.length).toFixed(3).padStart(13)}`,
  );
}
