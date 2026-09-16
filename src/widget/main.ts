import * as THREE from 'three';
import { generateCorridor } from '../sim/corridor';
import { createDrone, stepDrone } from '../sim/drone';
import { checkCollision } from '../sim/collisions';
import { CORRIDOR, DRONE_CONFIG, DT } from '../sim/episode';
import { createRenderer, createWorld } from '../render/scene';
import { createDroneCamera, captureFrame, horizontalFovDeg } from '../render/droneCamera';
import { createFlyController, stepFlyController } from '../controllers/flyCircuit';
import { createCnnController, loadCnnWeights, stepCnnController } from '../controllers/cnn';

const TRACE_SAMPLES = 240;
const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const controls = {
  rmo: el<HTMLInputElement>('rmo'),
  threshold: el<HTMLInputElement>('threshold'),
  speed: el<HTMLInputElement>('speed'),
  seed: el<HTMLSpanElement>('seed'),
};
const droneConfig = { ...DRONE_CONFIG };

interface Panel {
  view: HTMLCanvasElement;
  eye: CanvasRenderingContext2D;
  trace: CanvasRenderingContext2D;
  hits: HTMLElement;
  color: string;
  renderer: THREE.WebGLRenderer;
  chase: THREE.PerspectiveCamera;
  droneCam: ReturnType<typeof createDroneCamera>;
  scene: THREE.Scene;
  droneMesh: THREE.Mesh;
  drone: ReturnType<typeof createDrone>;
  samples: number[];
  collisions: number;
  lastHit: number | null;
}

function createPanel(key: 'fly' | 'cnn', color: number): Panel {
  const view = el<HTMLCanvasElement>(`view-${key}`);
  const renderer = createRenderer(view);
  const { scene, droneMesh } = createWorld([], CORRIDOR.length, CORRIDOR.width);
  (droneMesh.material as THREE.MeshStandardMaterial).color.set(color);
  return {
    view,
    eye: el<HTMLCanvasElement>(`eye-${key}`).getContext('2d') as CanvasRenderingContext2D,
    trace: el<HTMLCanvasElement>(`trace-${key}`).getContext('2d') as CanvasRenderingContext2D,
    hits: el(`hits-${key}`),
    color: `#${color.toString(16).padStart(6, '0')}`,
    renderer,
    chase: new THREE.PerspectiveCamera(60, 16 / 10, 0.1, 300),
    droneCam: createDroneCamera(),
    scene,
    droneMesh,
    drone: createDrone(),
    samples: [],
    collisions: 0,
    lastHit: null,
  };
}

const panels = { fly: createPanel('fly', 0x2f5fd0), cnn: createPanel('cnn', 0x6d4ed8) };

let seed = 1;
let obstacles = generateCorridor({ seed, ...CORRIDOR });
let flyController = createFlyController({});
let cnnController: ReturnType<typeof createCnnController> | null = null;

// Rebuilding the world is the only way to change corridors; the meshes are baked into the scene.
function loadCorridor(): void {
  obstacles = generateCorridor({ seed, ...CORRIDOR });
  controls.seed.textContent = String(seed);
  for (const panel of Object.values(panels)) {
    const { scene, droneMesh } = createWorld(obstacles, CORRIDOR.length, CORRIDOR.width);
    (droneMesh.material as THREE.MeshStandardMaterial).color.set(panel.color);
    panel.scene = scene;
    panel.droneMesh = droneMesh;
  }
  restart();
}

function restart(): void {
  for (const panel of Object.values(panels)) {
    panel.drone = createDrone();
    panel.samples = [];
    panel.collisions = 0;
    panel.lastHit = null;
    panel.hits.textContent = '0';
  }
  flyController = createFlyController({
    hFovDeg: horizontalFovDeg(panels.fly.droneCam.camera),
    vFovDeg: panels.fly.droneCam.camera.fov,
    rmoEnabled: controls.rmo.checked,
    escapeThreshold: Number(controls.threshold.value),
  });
  if (cnnController) cnnController = createCnnController(cnnController.weights);
}

function drawEye(panel: Panel, frame: Float32Array): void {
  const image = panel.eye.createImageData(64, 48);
  for (let i = 0; i < frame.length; i++) {
    const v = Math.round(frame[i] * 255);
    image.data.set([v, v, v, 255], i * 4);
  }
  panel.eye.putImageData(image, 0, 0);
}

// Traces share one drawing: fly plots GF drive against its threshold, CNN plots steering against zero.
function drawTrace(panel: Panel, value: number, top: number, bottom: number, marker: number): void {
  const { trace } = panel;
  if (trace.canvas.width !== trace.canvas.clientWidth) {
    trace.canvas.width = trace.canvas.clientWidth;
    trace.canvas.height = trace.canvas.clientHeight;
  }
  const { width, height } = trace.canvas;
  panel.samples.push(value);
  if (panel.samples.length > TRACE_SAMPLES) panel.samples.shift();

  const y = (v: number) => height - ((v - bottom) / (top - bottom)) * height;
  trace.clearRect(0, 0, width, height);
  trace.setLineDash([3, 3]);
  trace.strokeStyle = '#c8c8c2';
  trace.beginPath();
  trace.moveTo(0, y(marker));
  trace.lineTo(width, y(marker));
  trace.stroke();

  trace.setLineDash([]);
  trace.strokeStyle = panel.color;
  trace.lineWidth = 1.5;
  trace.beginPath();
  panel.samples.forEach((v, i) => {
    const px = (i / (TRACE_SAMPLES - 1)) * width;
    if (i === 0) trace.moveTo(px, y(v));
    else trace.lineTo(px, y(v));
  });
  trace.stroke();
}

function stepPanel(panel: Panel, steerFrom: (frame: Float32Array) => number): void {
  panel.droneMesh.visible = false;
  const frame = captureFrame(panel.renderer, panel.scene, panel.droneCam, panel.drone.x, panel.drone.z);
  panel.droneMesh.visible = true;
  drawEye(panel, frame);

  stepDrone(panel.drone, steerFrom(frame), DT, droneConfig);
  panel.drone.x = Math.max(-CORRIDOR.width / 2, Math.min(CORRIDOR.width / 2, panel.drone.x));

  const hit = checkCollision(panel.drone.x, panel.drone.z, obstacles);
  if (hit !== null && hit !== panel.lastHit) panel.hits.textContent = String(++panel.collisions);
  panel.lastHit = hit;
}

function render(panel: Panel): void {
  const { clientWidth, clientHeight } = panel.view;
  if (panel.view.width !== clientWidth || panel.view.height !== clientHeight) {
    panel.renderer.setSize(clientWidth, clientHeight, false);
    panel.chase.aspect = clientWidth / clientHeight;
    panel.chase.updateProjectionMatrix();
  }
  panel.droneMesh.position.set(panel.drone.x, 0.6, panel.drone.z);
  panel.chase.position.set(panel.drone.x, 2.2, panel.drone.z - 6);
  panel.chase.lookAt(panel.drone.x, 0.6, panel.drone.z + 5);
  panel.renderer.render(panel.scene, panel.chase);
}

function frame(): void {
  const { fly, cnn } = panels;

  let gfDrive = 0;
  stepPanel(fly, (frame) => {
    const step = stepFlyController(flyController, frame, DT);
    gfDrive = step.debug.gfDrive;
    return step.steering;
  });
  drawTrace(fly, gfDrive, Math.max(2.5, flyController.config.escapeThreshold * 1.5), 0, flyController.config.escapeThreshold);

  stepPanel(cnn, (frame) => (cnnController ? stepCnnController(cnnController, frame, DT) : 0));
  drawTrace(cnn, cnnController ? cnnController.steering : 0, 1, -1, 0);

  render(fly);
  render(cnn);

  if (Math.min(fly.drone.z, cnn.drone.z) >= CORRIDOR.length) {
    seed++;
    loadCorridor();
  }
}

function tick(): void {
  frame();
  requestAnimationFrame(tick);
}

el('restart').onclick = restart;
el('next-seed').onclick = () => {
  seed++;
  loadCorridor();
};
controls.rmo.onchange = restart;
controls.threshold.oninput = () => {
  el('threshold-val').textContent = Number(controls.threshold.value).toFixed(2);
  flyController.config.escapeThreshold = Number(controls.threshold.value);
};
controls.speed.oninput = () => {
  el('speed-val').textContent = Number(controls.speed.value).toFixed(1);
  droneConfig.speed = Number(controls.speed.value);
};
// The CSS drops to one column on narrow screens, so start on the fly panel and let the button swap them.
if (window.matchMedia('(max-width: 720px)').matches) el('panel-cnn').classList.add('hidden');

el('swap').onclick = () => {
  const showingFly = !el('panel-fly').classList.contains('hidden');
  el('panel-fly').classList.toggle('hidden', showingFly);
  el('panel-cnn').classList.toggle('hidden', !showingFly);
  el('swap').textContent = showingFly ? 'Show fly circuit' : 'Show CNN';
};

Object.assign(window, { panels, frame }); // ponytail: debug handle, drop before shipping the widget

fetch('cnn.bin')
  .then((r) => r.arrayBuffer())
  .then((b) => {
    cnnController = createCnnController(loadCnnWeights(new Uint8Array(b)));
  });

loadCorridor();
requestAnimationFrame(tick);
