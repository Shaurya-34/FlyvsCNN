// Episode tasks shared by the Node scripts (harness renderer) and the headless benchmark page (real Three.js renderer).
import { createCnnController, stepCnnController, type CnnWeights } from '../controllers/cnn';
import { createFlyController, stepFlyController, type FlyControllerConfig } from '../controllers/flyCircuit';
import { mulberry32 } from '../sim/corridor';
import { DT, flyEpisode, type Renderer, type Steer } from '../sim/episode';
import { expertSteer } from '../sim/expert';

const RECORD_EVERY = 3; // 60Hz sim -> 20Hz frames. The CNN's 4-frame stack must use this same spacing at inference.

export function recordExpertEpisode(seed: number, renderer: Renderer) {
  const rng = mulberry32(seed + 1_000_000); // separate stream from the corridor layout
  const frames: Uint8Array[] = [];
  const steering: number[] = [];
  let noise = 0;
  let step = 0;

  const { collisions } = flyEpisode(
    seed,
    (frame, drone, obstacles) => {
      const label = expertSteer(drone, obstacles);
      if (step % RECORD_EVERY === 0) {
        frames.push(Uint8Array.from(frame, (v) => Math.round(v * 255)));
        steering.push(label);
      }
      // Occasional steering noise so the data includes recovering from bad positions. Labels stay clean.
      if (step % 30 === 0) noise = rng() < 0.3 ? (rng() - 0.5) * 1.2 : 0;
      step++;
      return Math.max(-1, Math.min(1, label + noise));
    },
    renderer,
  );

  const all = new Uint8Array(frames.length * frames[0].length);
  frames.forEach((f, i) => all.set(f, i * f.length));
  return { frames: all, steering: Float32Array.from(steering), collisions };
}

// Per-step record of what the fly circuit saw and did, for comparing its size estimate against the real geometry.
// Columns: droneX, droneZ, sizeDeg, gfDrive, escaping.
export const TRACE_COLUMNS = 5;

export function traceEpisode(seed: number, tuning: Partial<FlyControllerConfig>, renderer: Renderer): Float32Array {
  const controller = createFlyController(tuning);
  const rows: number[] = [];
  flyEpisode(
    seed,
    (frame, drone) => {
      const { steering, debug } = stepFlyController(controller, frame, DT);
      rows.push(drone.x, drone.z, debug.sizeDeg, debug.gfDrive, debug.escaping ? 1 : 0);
      return steering;
    },
    renderer,
  );
  return Float32Array.from(rows);
}

export type ContestantSpec =
  | { kind: 'fly'; drive: boolean; tuning: Partial<FlyControllerConfig> }
  | { kind: 'expert' }
  | { kind: 'cnn'; path: string };

export function benchmarkEpisode(
  seed: number,
  spec: ContestantSpec,
  renderer: Renderer,
  weightsFor: (path: string) => CnnWeights,
) {
  let escapes = 0;
  let effort = 0;
  let steps = 0;
  const track = (steering: number) => {
    effort += Math.abs(steering);
    steps++;
    return steering;
  };

  let steer: Steer;
  if (spec.kind === 'fly') {
    const controller = createFlyController(spec.tuning);
    steer = (frame) => {
      const { steering, debug } = stepFlyController(controller, frame, DT);
      if (debug.escaping) escapes++;
      track(steering);
      return spec.drive ? steering : 0;
    };
  } else if (spec.kind === 'expert') {
    steer = (_frame, drone, obstacles) => track(expertSteer(drone, obstacles));
  } else {
    const controller = createCnnController(weightsFor(spec.path));
    steer = (frame) => track(stepCnnController(controller, frame, DT));
  }

  const { collisions } = flyEpisode(seed, steer, renderer);
  return { collisions, escapes, effort, steps };
}
