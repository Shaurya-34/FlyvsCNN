// Fly circuit: LPLC2 (size) + LC4 (rate) + two size-dependent inhibitory inputs -> GF drive, Ache et al. 2019
// Curr Biol, STAR Methods eq 3-7. RMO compares flow magnitude to an eccentricity-scaled self-motion baseline.

const OMM_COLS = 16;
const OMM_ROWS = 12;
const CELLS = OMM_COLS * OMM_ROWS;
const FOE_X = (OMM_COLS - 1) / 2;
const FOE_Y = (OMM_ROWS - 1) / 2;
const NOISE_FLOOR = 0.01;
const CALIBRATION_MIN_ECCENTRICITY = 4.0; // near-FOE ratios are noise, so fit the slope from outer cells only

// Published constants, never tuned. The paper's sensory delays (11-37.5 ms) are not modeled.
const LC4_C1 = 0.0002567;
const LPLC2_C2 = 1.7;
const LPLC2_C3 = 42; // degrees
const LPLC2_C4 = 0.52;
const I1_C5 = -0.53; // tonic inhibition, a sigmoid in size (eq 5)
const I1_C6 = 0.59;
const I1_C7 = 66; // degrees
const I1_C8 = -11;
const I2_C9 = -0.52; // LC4-dependent inhibition, a Gaussian dip in size (eq 6)
const I2_C10 = 26; // degrees
const I2_C11 = 7.8;
const W_LPLC2 = 1.45;
const W_LC4 = 1.62;
const W_I1 = 2.27;
const W_I2 = 1;

function vLplc2(sizeDeg: number): number {
  const theta = Math.max(sizeDeg, 1e-3);
  const d = Math.log(theta) - Math.log(LPLC2_C3);
  return LPLC2_C2 * Math.exp(-(d * d) / (2 * LPLC2_C4 * LPLC2_C4));
}

function inhibition(sizeDeg: number): number {
  const tonic = I1_C5 + I1_C6 / (1 + Math.exp(-(sizeDeg - I1_C7) / I1_C8));
  const dip = I2_C9 * Math.exp(-((sizeDeg - I2_C10) ** 2) / (2 * I2_C11 * I2_C11));
  return W_I1 * tonic + W_I2 * dip;
}

// Escape threshold rule: half the peak of the model's own size tuning, for a disc held still (so no LC4 term).
const HALF_MAX_DRIVE =
  0.5 * Math.max(...Array.from({ length: 180 }, (_, i) => W_LPLC2 * vLplc2(i + 1) + inhibition(i + 1)));

// Engineering constants, not biology. Defaults are the hand-picked values; scripts/tune-fly.ts searches them.
export const DEFAULT_TUNING = {
  // Fire where the published model's size tuning is at half maximum (about 1.2, a ~26 deg object). A rule read
  // off the biology, not a number fitted to our corridors.
  escapeThreshold: HALF_MAX_DRIVE,
  steeringGain: 1,
  evasiveDurationS: 0.3,
  hysteresis: 0.8,
  excessMargin: 1.5,
  componentRatio: 0.35,
  gradientFloorSq: 0.005, // normal flow blows up on near-flat cells
};

export type FlyTuning = typeof DEFAULT_TUNING;

const ECCENTRICITY = Float32Array.from({ length: CELLS }, (_, i) =>
  Math.hypot((i % OMM_COLS) - FOE_X, ((i / OMM_COLS) | 0) - FOE_Y),
);

export interface FlyControllerConfig extends FlyTuning {
  hFovDeg: number;
  vFovDeg: number;
  rmoEnabled: boolean;
  inhibitionEnabled: boolean;
}

interface FlyControllerState {
  prevOmm: Float32Array | null;
  prevSizeDeg: number;
  armed: boolean;
  evasiveUntil: number;
  simTime: number;
}

export interface FlyControllerDebug {
  gfDrive: number;
  sizeDeg: number;
  rateDegPerSec: number;
  escaping: boolean;
}

export interface FlyController {
  config: FlyControllerConfig;
  state: FlyControllerState;
}

export function createFlyController(config: Partial<FlyControllerConfig> = {}): FlyController {
  return {
    config: { hFovDeg: 90, vFovDeg: 90, rmoEnabled: true, inhibitionEnabled: true, ...DEFAULT_TUNING, ...config },
    state: {
      prevOmm: null,
      prevSizeDeg: 0,
      armed: true,
      evasiveUntil: -Infinity,
      simTime: 0,
    },
  };
}

// Input is always 64x48, so each ommatidium is an exact 4x4 block.
function downsampleToOmmatidia(frame: Float32Array): Float32Array {
  const omm = new Float32Array(CELLS);
  for (let i = 0; i < frame.length; i++) {
    omm[(((i / 64) | 0) >> 2) * OMM_COLS + ((i % 64) >> 2)] += frame[i];
  }
  for (let i = 0; i < CELLS; i++) omm[i] /= 16;
  return omm;
}

function at(grid: Float32Array, x: number, y: number): number {
  const cx = Math.min(Math.max(x, 0), OMM_COLS - 1);
  const cy = Math.min(Math.max(y, 0), OMM_ROWS - 1);
  return grid[cy * OMM_COLS + cx];
}

// The model's theta is ONE object's angular size, so read it from the largest contiguous region, not a sum over the eye.
function largestActiveComponent(
  rawSignal: Float32Array,
  threshold: number,
): { cellCount: number; centroidX: number; width: number; height: number } {
  const visited = new Uint8Array(CELLS);
  const stack: number[] = [];
  let best = { cellCount: 0, centroidX: FOE_X, width: 0, height: 0 };

  for (let start = 0; start < CELLS; start++) {
    if (visited[start] || rawSignal[start] <= threshold) continue;
    stack.length = 0;
    stack.push(start);
    visited[start] = 1;
    let cellCount = 0;
    let sumX = 0;
    let minX = OMM_COLS;
    let maxX = -1;
    let minY = OMM_ROWS;
    let maxY = -1;

    while (stack.length > 0) {
      const i = stack.pop() as number;
      const x = i % OMM_COLS;
      const y = (i / OMM_COLS) | 0;
      cellCount++;
      sumX += x;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= OMM_COLS || ny >= OMM_ROWS) continue;
          const ni = ny * OMM_COLS + nx;
          if (visited[ni] || rawSignal[ni] <= threshold) continue;
          visited[ni] = 1;
          stack.push(ni);
        }
      }
    }

    if (cellCount > best.cellCount) {
      best = { cellCount, centroidX: sumX / cellCount, width: maxX - minX + 1, height: maxY - minY + 1 };
    }
  }
  return best;
}

export function stepFlyController(
  controller: FlyController,
  frame: Float32Array,
  dt: number,
): { steering: number; debug: FlyControllerDebug } {
  const { config, state } = controller;
  state.simTime += dt;

  const omm = downsampleToOmmatidia(frame);
  if (!state.prevOmm) {
    state.prevOmm = omm;
    return { steering: 0, debug: { gfDrive: 0, sizeDeg: 0, rateDegPerSec: 0, escaping: false } };
  }
  const prevOmm = state.prevOmm;

  // Normal-flow magnitude |dI/dt| / |grad I|.
  const magnitude = new Float32Array(CELLS);
  for (let y = 0; y < OMM_ROWS; y++) {
    for (let x = 0; x < OMM_COLS; x++) {
      const i = y * OMM_COLS + x;
      const gx = (at(omm, x + 1, y) - at(omm, x - 1, y)) / 2;
      const gy = (at(omm, x, y + 1) - at(omm, x, y - 1)) / 2;
      const gradMagSq = gx * gx + gy * gy;
      if (gradMagSq < config.gradientFloorSq) continue;
      magnitude[i] = Math.abs((omm[i] - prevOmm[i]) / dt) / Math.sqrt(gradMagSq);
    }
  }

  const rawSignal = new Float32Array(CELLS);
  if (config.rmoEnabled) {
    const ratios: number[] = [];
    for (let i = 0; i < CELLS; i++) {
      if (magnitude[i] > NOISE_FLOOR && ECCENTRICITY[i] >= CALIBRATION_MIN_ECCENTRICITY) {
        ratios.push(magnitude[i] / ECCENTRICITY[i]);
      }
    }
    ratios.sort((a, b) => a - b);
    const slopeK = ratios.length > 0 ? ratios[Math.floor(ratios.length / 2)] : 0;
    for (let i = 0; i < CELLS; i++) {
      rawSignal[i] = Math.max(0, magnitude[i] - slopeK * ECCENTRICITY[i] * config.excessMargin);
    }
  } else {
    rawSignal.set(magnitude);
  }

  // Threshold relative to this frame's peak, since flow scale varies with dt and speed.
  const object = largestActiveComponent(rawSignal, Math.max(NOISE_FLOOR, config.componentRatio * Math.max(...rawSignal)));

  // Flat-shaded surfaces only produce flow at their edges, so the component is a ribbon and its cell count badly
  // under-reads a real obstacle. Obstacles are solid, so take the bounding box as filled and report the diameter of
  // the circle with that area: a near pole grows properly, while a floor-grid streak stays thin and small.
  const widthDeg = (object.width * config.hFovDeg) / OMM_COLS;
  const heightDeg = (object.height * config.vFovDeg) / OMM_ROWS;
  const sizeDeg = 2 * Math.sqrt((widthDeg * heightDeg) / Math.PI);
  const rateDegPerSec = (sizeDeg - state.prevSizeDeg) / dt;
  const gfDrive =
    W_LPLC2 * vLplc2(sizeDeg) +
    W_LC4 * (LC4_C1 * rateDegPerSec) +
    (config.inhibitionEnabled ? inhibition(sizeDeg) : 0);

  let escaping = state.simTime < state.evasiveUntil;
  if (state.armed && gfDrive >= config.escapeThreshold) {
    state.evasiveUntil = state.simTime + config.evasiveDurationS;
    state.armed = false;
    escaping = true;
  } else if (gfDrive < config.escapeThreshold * config.hysteresis) {
    state.armed = true;
  }

  const direction = object.centroidX < FOE_X ? 1 : -1; // dodge away from the threat's side
  const continuousSteering = Math.max(-1, Math.min(1, (config.steeringGain * direction * gfDrive) / config.escapeThreshold));
  const steering = escaping ? direction : continuousSteering;

  state.prevOmm = omm;
  state.prevSizeDeg = sizeDeg;

  return { steering, debug: { gfDrive, sizeDeg, rateDegPerSec, escaping } };
}
