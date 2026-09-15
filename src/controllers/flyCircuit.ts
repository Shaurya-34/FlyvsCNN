// Fly circuit: LPLC2 (size) + LC4 (rate) -> GF drive, Ache et al. 2019 Curr Biol, STAR Methods eq 3, 4, 7
// (inhibitory eq 5-6 dropped). RMO compares flow magnitude to an eccentricity-scaled self-motion baseline.

const OMM_COLS = 16;
const OMM_ROWS = 12;
const CELLS = OMM_COLS * OMM_ROWS;
const FOE_X = (OMM_COLS - 1) / 2;
const FOE_Y = (OMM_ROWS - 1) / 2;
const NOISE_FLOOR = 0.01;
const CALIBRATION_MIN_ECCENTRICITY = 4.0; // near-FOE ratios are noise, so fit the slope from outer cells only

// Published constants, never tuned. The paper's ~19ms sensory delay is not modeled.
const LC4_C1 = 0.0002567;
const LPLC2_C2 = 1.7;
const LPLC2_C3 = 42; // degrees
const LPLC2_C4 = 0.52;
const W_LPLC2 = 1.45;
const W_LC4 = 1.62;

// Engineering constants, not biology. Defaults are the hand-picked values; scripts/tune-fly.ts searches them.
export const DEFAULT_TUNING = {
  escapeThreshold: 1.0,
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
    config: { hFovDeg: 90, vFovDeg: 90, rmoEnabled: true, ...DEFAULT_TUNING, ...config },
    state: {
      prevOmm: null,
      prevSizeDeg: 0,
      armed: true,
      evasiveUntil: -Infinity,
      simTime: 0,
    },
  };
}

function vLplc2(sizeDeg: number): number {
  const theta = Math.max(sizeDeg, 1e-3);
  const d = Math.log(theta) - Math.log(LPLC2_C3);
  return LPLC2_C2 * Math.exp(-(d * d) / (2 * LPLC2_C4));
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
function largestActiveComponent(rawSignal: Float32Array, threshold: number): { cellCount: number; centroidX: number } {
  const visited = new Uint8Array(CELLS);
  const stack: number[] = [];
  let best = { cellCount: 0, centroidX: FOE_X };

  for (let start = 0; start < CELLS; start++) {
    if (visited[start] || rawSignal[start] <= threshold) continue;
    stack.length = 0;
    stack.push(start);
    visited[start] = 1;
    let cellCount = 0;
    let sumX = 0;

    while (stack.length > 0) {
      const i = stack.pop() as number;
      const x = i % OMM_COLS;
      const y = (i / OMM_COLS) | 0;
      cellCount++;
      sumX += x;

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

    if (cellCount > best.cellCount) best = { cellCount, centroidX: sumX / cellCount };
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

  const avgCellDeg = (config.hFovDeg / OMM_COLS + config.vFovDeg / OMM_ROWS) / 2;
  const sizeDeg = 2 * Math.sqrt(object.cellCount / Math.PI) * avgCellDeg; // equivalent-circle diameter
  const rateDegPerSec = (sizeDeg - state.prevSizeDeg) / dt;
  const gfDrive = W_LPLC2 * vLplc2(sizeDeg) + W_LC4 * (LC4_C1 * rateDegPerSec);

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
