import type { Obstacle } from './corridor';

const DRONE_RADIUS = 0.3;

// Loose circle-vs-circle check in the xz-plane, not exact physics.
export function checkCollision(x: number, z: number, obstacles: Obstacle[]): number | null {
  for (let i = 0; i < obstacles.length; i++) {
    const o = obstacles[i];
    const r = o.type === 'pole' ? o.radius! : Math.hypot(o.width!, o.depth!) / 2;
    if (Math.hypot(x - o.x, z - o.z) < r + DRONE_RADIUS) {
      return i;
    }
  }
  return null;
}
