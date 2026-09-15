import type { Obstacle } from './corridor';

export const DRONE_RADIUS = 0.3;

export function obstacleRadius(o: Obstacle): number {
  return o.type === 'pole' ? o.radius! : Math.hypot(o.width!, o.depth!) / 2;
}

// Loose circle-vs-circle check in the xz-plane, not exact physics.
export function checkCollision(x: number, z: number, obstacles: Obstacle[]): number | null {
  for (let i = 0; i < obstacles.length; i++) {
    const o = obstacles[i];
    if (Math.hypot(x - o.x, z - o.z) < obstacleRadius(o) + DRONE_RADIUS) {
      return i;
    }
  }
  return null;
}
