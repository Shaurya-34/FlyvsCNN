import { generateCorridor, type Obstacle } from './corridor';
import { createDrone, stepDrone, type DroneState } from './drone';
import { checkCollision } from './collisions';

export const CORRIDOR = { length: 120, width: 8, density: 0.3, sizeVariance: 0.4 };
export const DRONE_CONFIG = { speed: 4, lateralDamping: 2 };
export const DT = 1 / 60;

export type Render = (droneX: number, droneZ: number) => Float32Array;
export type Renderer = (obstacles: Obstacle[]) => Render; // builds a renderer for one corridor
export type Steer = (frame: Float32Array, drone: DroneState, obstacles: Obstacle[]) => number;

// One full corridor run. Controllers that must not cheat should only read `frame`.
export function flyEpisode(seed: number, steer: Steer, renderer: Renderer): { collisions: number } {
  const obstacles = generateCorridor({ seed, ...CORRIDOR });
  const render = renderer(obstacles);
  const drone = createDrone();
  let collisions = 0;
  let lastHit: number | null = null;

  while (drone.z < CORRIDOR.length) {
    const steering = steer(render(drone.x, drone.z), drone, obstacles);
    stepDrone(drone, steering, DT, DRONE_CONFIG);
    drone.x = Math.max(-CORRIDOR.width / 2, Math.min(CORRIDOR.width / 2, drone.x));

    const hit = checkCollision(drone.x, drone.z, obstacles);
    if (hit !== null && hit !== lastHit) collisions++;
    lastHit = hit;
  }
  return { collisions };
}
