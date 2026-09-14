/**
 * Headless stand-in for the Three.js render, so the controller can be tested
 * against the REAL corridor geometry and REAL drone kinematics without a
 * WebGL context. Projects obstacle silhouettes and ground grid lines with a
 * plain pinhole camera -- which is close to what the actual flat-shaded
 * low-poly scene looks like from the drone's viewpoint (grey rectangles on a
 * pale ground, plus grid lines that supply self-motion optical flow).
 *
 * Not a replacement for verifying in the browser, but it makes the
 * controller testable in a fast loop.
 */
import type { Obstacle } from '../src/sim/corridor';

export const W = 64;
export const H = 48;
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
