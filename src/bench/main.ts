// Headless benchmark page (bench.html): runs episode tasks on the real Three.js renderer for scripts/runner.ts.
import * as THREE from 'three';
import { loadCnnWeights, type CnnWeights } from '../controllers/cnn';
import { captureFrame, createDroneCamera } from '../render/droneCamera';
import { createRenderer, createWorld } from '../render/scene';
import { CORRIDOR, type Renderer } from '../sim/episode';
import { benchmarkEpisode, recordExpertEpisode, traceEpisode, type ContestantSpec } from './tasks';
import type { FlyControllerConfig } from '../controllers/flyCircuit';

const renderer = createRenderer(document.createElement('canvas'));
const droneCam = createDroneCamera();
let world: THREE.Scene | null = null;

const threeRenderer: Renderer = (obstacles) => {
  if (world) dispose(world);
  const { scene, droneMesh } = createWorld(obstacles, CORRIDOR.length, CORRIDOR.width);
  droneMesh.visible = false;
  world = scene;
  return (droneX, droneZ) => captureFrame(renderer, scene, droneCam, droneX, droneZ);
};

function dispose(scene: THREE.Scene): void {
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose();
    (mesh.material as THREE.Material | undefined)?.dispose();
  });
}

const base64 = (a: ArrayBufferView) => {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
};

const weights = new Map<string, CnnWeights>();

Object.assign(window, {
  traceEpisode(seed: number, tuning: Partial<FlyControllerConfig>) {
    return base64(traceEpisode(seed, tuning, threeRenderer));
  },
  recordExpertEpisode(seed: number) {
    const r = recordExpertEpisode(seed, threeRenderer);
    return { frames: base64(r.frames), steering: base64(r.steering), collisions: r.collisions };
  },
  async benchmarkEpisode(seed: number, spec: ContestantSpec) {
    if (spec.kind === 'cnn' && !weights.has(spec.path)) {
      const res = await fetch(`/${spec.path.replace(/^public\//, '')}`); // Vite serves public/ at the root
      weights.set(spec.path, loadCnnWeights(new Uint8Array(await res.arrayBuffer())));
    }
    return benchmarkEpisode(seed, spec, threeRenderer, (path) => weights.get(path) as CnnWeights);
  },
});
