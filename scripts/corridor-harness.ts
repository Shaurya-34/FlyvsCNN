/**
 * Headless stand-in for the Three.js render, so controllers can be tested
 * against the REAL corridor geometry and REAL drone kinematics without a
 * WebGL context. Projects obstacle silhouettes and ground grid lines with a
 * plain pinhole camera, which is close to what the flat-shaded scene looks
 * like from the drone's viewpoint. Its pixels are not identical to Three.js.
 */
import { generateCorridor, type Obstacle } from '../src/sim/corridor';
import { createDrone, stepDrone, type DroneState } from '../src/sim/drone';
import { checkCollision } from '../src/sim/collisions';

export const W = 64;
export const H = 48;
export const CORRIDOR = { length: 120, width: 8, density: 0.3, sizeVariance: 0.4 };
export const DRONE_CONFIG = { speed: 4, lateralDamping: 2 };
export const DT = 1 / 60;

const CAM_Y = 0.6;
const V_FOV_DEG = 90;

const vFovRad = (V_FOV_DEG * Math.PI) / 180;
const fv = 1 / Math.tan(vFovRad / 2);
const aspect = W / H;
const fh = fv / aspect;

export function renderFrame(droneX: number, droneZ: number, obstacles: Obstacle[]): Float32Array {
  const out = new Float32Array(W * H).fill(0.93);

  const horizon = Math.round(H / 2);
  for (let y = horizon; y < H; y++) {
    for (let x = 0; x < W; x++) out[y * W + x] = 0.85;
  }

  // Transverse ground grid lines -- the self-motion optical flow source.
  for (let gz = Math.ceil(droneZ); gz < droneZ + 40; gz++) {
    const dz = gz - droneZ;
    if (dz < 0.3) continue;
    const ndcY = (-CAM_Y / dz) * fv;
    const py = Math.round(((1 - ndcY) / 2) * H);
    if (py >= 0 && py < H) {
      for (let x = 0; x < W; x++) out[py * W + x] = 0.72;
    }
  }

  // Obstacles, far to near so nearer ones paint over farther ones.
  const visible = obstacles
    .map((o) => ({ o, dz: o.z - droneZ, dx: o.x - droneX }))
    .filter((v) => v.dz > 0.25 && v.dz < 60)
    .sort((a, b) => b.dz - a.dz);

  for (const { o, dz, dx } of visible) {
    const halfWidth = o.type === 'pole' ? (o.radius as number) : (o.width as number) / 2;

    const ndcL = ((dx - halfWidth) / dz) * fh;
    const ndcR = ((dx + halfWidth) / dz) * fh;
    const ndcBottom = (-CAM_Y / dz) * fv;
    const ndcTop = ((o.height - CAM_Y) / dz) * fv;

    const px0 = Math.round(((ndcL + 1) / 2) * W);
    const px1 = Math.round(((ndcR + 1) / 2) * W);
    const py0 = Math.round(((1 - ndcTop) / 2) * H);
    const py1 = Math.round(((1 - ndcBottom) / 2) * H);

    for (let y = Math.max(0, py0); y <= Math.min(H - 1, py1); y++) {
      for (let x = Math.max(0, px0); x <= Math.min(W - 1, px1); x++) {
        out[y * W + x] = 0.45;
      }
    }
  }

  return out;
}

export type Steer = (frame: Float32Array, drone: DroneState, obstacles: Obstacle[]) => number;

// One full corridor run. Controllers that must not cheat should only read `frame`.
export function flyEpisode(seed: number, steer: Steer): { collisions: number } {
  const obstacles = generateCorridor({ seed, ...CORRIDOR });
  const drone = createDrone();
  let collisions = 0;
  let lastHit: number | null = null;

  while (drone.z < CORRIDOR.length) {
    const steering = steer(renderFrame(drone.x, drone.z, obstacles), drone, obstacles);
    stepDrone(drone, steering, DT, DRONE_CONFIG);
    drone.x = Math.max(-CORRIDOR.width / 2, Math.min(CORRIDOR.width / 2, drone.x));

    const hit = checkCollision(drone.x, drone.z, obstacles);
    if (hit !== null && hit !== lastHit) collisions++;
    lastHit = hit;
  }
  return { collisions };
}
