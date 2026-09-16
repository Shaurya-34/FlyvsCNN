import type { Obstacle } from './corridor';
import { stepDrone, type DroneState } from './drone';
import { DRONE_RADIUS, obstacleRadius } from './collisions';
import { CORRIDOR, DRONE_CONFIG } from './episode';

const LOOKAHEAD_S = 2;
const ROLLOUT_DT = 0.05;
const MARGIN = 0.35;
const HALF_LANE = CORRIDOR.width / 2 - 0.5;

const pd = (d: DroneState, targetX: number) => Math.max(-1, Math.min(1, 1.5 * (targetX - d.x) - d.lateralVel));

// CHEATS on purpose: reads obstacle positions and drone velocity. Only used to label CNN training data, never as a
// contestant. Rolls out steering toward each candidate lateral position with the real kinematics and picks the path
// that overlaps obstacles least (scoring only the destination steered it straight through obstacles in the way).
export function expertSteer(drone: DroneState, obstacles: Obstacle[]): number {
  const horizonZ = drone.z + DRONE_CONFIG.speed * LOOKAHEAD_S + 2;
  const nearby = obstacles.filter((o) => o.z > drone.z - 2 && o.z < horizonZ);

  let bestX = drone.x;
  let bestCost = Infinity;
  for (let cx = -HALF_LANE; cx <= HALF_LANE; cx += 0.25) {
    const sim = { ...drone };
    let cost = 0.1 * Math.abs(cx - drone.x);
    for (let t = 0; t < LOOKAHEAD_S; t += ROLLOUT_DT) {
      stepDrone(sim, pd(sim, cx), ROLLOUT_DT, DRONE_CONFIG);
      for (const o of nearby) {
        const overlap = obstacleRadius(o) + DRONE_RADIUS + MARGIN - Math.hypot(sim.x - o.x, sim.z - o.z);
        if (overlap > 0) cost += overlap;
      }
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestX = cx;
    }
  }
  return pd(drone, bestX);
}
