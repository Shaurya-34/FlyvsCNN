// The widget, as a self-contained block: it builds its own markup and styles inside whatever element it is given,
// so a host page only needs `<figure data-fly-vs-cnn></figure>` and this bundle. `data-weights` overrides where the
// CNN weights are fetched from, which matters when the host serves them from somewhere other than the page's folder.
import * as THREE from 'three';
import { generateCorridor } from '../sim/corridor';
import { createDrone, stepDrone } from '../sim/drone';
import { checkCollision } from '../sim/collisions';
import { CORRIDOR, DRONE_CONFIG, DT } from '../sim/episode';
import { createRenderer, createWorld } from '../render/scene';
import { createDroneCamera, captureFrameAsync, horizontalFovDeg } from '../render/droneCamera';
import { createFlyController, stepFlyController } from '../controllers/flyCircuit';
import { createCnnController, loadCnnWeights, stepCnnController } from '../controllers/cnn';

const TRACE_SAMPLES = 240;
const FLY_COLOR = 0x2f5fd0;
const CNN_COLOR = 0x6d4ed8;

const STYLE = `
.fvc { --fvc-ink: #14161a; --fvc-muted: #6b7280; --fvc-line: #d9d9d4; --fvc-panel: #fff;
  --fvc-fly: #2f5fd0; --fvc-cnn: #6d4ed8; margin: 0; color: var(--fvc-ink); }
@media (prefers-color-scheme: dark) {
  .fvc { --fvc-ink: #e8e8e4; --fvc-muted: #9aa0aa; --fvc-line: #34363b; --fvc-panel: #1b1d21;
    --fvc-fly: #7aa2f7; --fvc-cnn: #a48cf0; }
}
[data-theme="dark"] .fvc { --fvc-ink: #e8e8e4; --fvc-muted: #9aa0aa; --fvc-line: #34363b; --fvc-panel: #1b1d21;
  --fvc-fly: #7aa2f7; --fvc-cnn: #a48cf0; }
[data-theme="light"] .fvc { --fvc-ink: #14161a; --fvc-muted: #6b7280; --fvc-line: #d9d9d4; --fvc-panel: #fff;
  --fvc-fly: #2f5fd0; --fvc-cnn: #6d4ed8; }
.fvc * { box-sizing: border-box; }
.fvc-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.fvc-panel { min-width: 0; }
.fvc-panel.is-hidden { display: none; }
.fvc-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.fvc-name { font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; }
.fvc-panel[data-who="fly"] .fvc-name { color: var(--fvc-fly); }
.fvc-panel[data-who="cnn"] .fvc-name { color: var(--fvc-cnn); }
.fvc-note { color: var(--fvc-muted); font-size: 12px; }
.fvc-hits { font-variant-numeric: tabular-nums; }
.fvc-stage { position: relative; aspect-ratio: 16 / 10; border: 1px solid var(--fvc-line); background: var(--fvc-panel); }
.fvc-view { width: 100%; height: 100%; display: block; }
.fvc-eye { position: absolute; left: 8px; bottom: 8px; width: 128px; height: 96px; max-width: 45%;
  image-rendering: pixelated; border: 1px solid var(--fvc-line); background: var(--fvc-panel); }
.fvc-eye-label { position: absolute; left: 8px; bottom: 108px; padding: 1px 5px; font-size: 11px;
  color: var(--fvc-muted); background: color-mix(in srgb, var(--fvc-panel) 85%, transparent); }
.fvc-trace { width: 100%; height: 72px; display: block; border: 1px solid var(--fvc-line); border-top: 0;
  background: var(--fvc-panel); }
.fvc-trace-label { margin-top: 6px; font-size: 12px; color: var(--fvc-muted); }
.fvc-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 18px;
  padding-top: 14px; border-top: 1px solid var(--fvc-line); }
.fvc-controls label { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.fvc-controls input[type="range"] { width: 120px; accent-color: var(--fvc-fly); }
.fvc-controls input[type="checkbox"] { accent-color: var(--fvc-fly); }
.fvc button { font: inherit; font-size: 13px; padding: 5px 12px; color: var(--fvc-ink); background: var(--fvc-panel);
  border: 1px solid var(--fvc-line); border-radius: 2px; cursor: pointer; }
.fvc button:hover { border-color: var(--fvc-muted); }
.fvc-val { color: var(--fvc-muted); font-variant-numeric: tabular-nums; min-width: 3.2em; }
.fvc-swap, .fvc-tap { display: none; }
.fvc-tap { position: absolute; right: 8px; top: 8px; padding: 1px 6px; font-size: 11px; color: var(--fvc-muted);
  background: color-mix(in srgb, var(--fvc-panel) 85%, transparent); pointer-events: none; }
@media (max-width: 720px) {
  .fvc-panels { grid-template-columns: 1fr; }
  .fvc-swap, .fvc-tap { display: inline-flex; }
  .fvc-stage { cursor: pointer; -webkit-tap-highlight-color: transparent; }
}
`;

const MARKUP = `
<div class="fvc-panels">
  <section class="fvc-panel" data-who="fly">
    <div class="fvc-head">
      <span class="fvc-name">Fly circuit</span>
      <span class="fvc-note">zero training &middot; <span class="fvc-hits" data-hits>0</span> hits</span>
    </div>
    <div class="fvc-stage">
      <canvas class="fvc-view" data-view></canvas>
      <span class="fvc-eye-label">what it sees</span>
      <canvas class="fvc-eye" data-eye width="64" height="48"></canvas>
      <span class="fvc-tap">tap for the CNN</span>
    </div>
    <canvas class="fvc-trace" data-trace></canvas>
    <div class="fvc-trace-label">Giant Fiber drive, with escape threshold</div>
  </section>
  <section class="fvc-panel" data-who="cnn">
    <div class="fvc-head">
      <span class="fvc-name">CNN</span>
      <span class="fvc-note">200 training episodes &middot; <span class="fvc-hits" data-hits>0</span> hits</span>
    </div>
    <div class="fvc-stage">
      <canvas class="fvc-view" data-view></canvas>
      <span class="fvc-eye-label">what it sees</span>
      <canvas class="fvc-eye" data-eye width="64" height="48"></canvas>
      <span class="fvc-tap">tap for the fly circuit</span>
    </div>
    <canvas class="fvc-trace" data-trace></canvas>
    <div class="fvc-trace-label">steering output</div>
  </section>
</div>
<div class="fvc-controls">
  <button type="button" data-restart>Restart</button>
  <button type="button" data-next-seed>New corridor</button>
  <span class="fvc-note">corridor <span data-seed>1</span></span>
  <label><input type="checkbox" data-rmo checked> motion opponency</label>
  <label>escape threshold
    <input type="range" data-threshold min="0.3" max="2.5" step="0.01" value="1.23">
    <span class="fvc-val" data-threshold-val>1.23</span>
  </label>
  <label>speed
    <input type="range" data-speed min="1" max="8" step="0.5" value="4">
    <span class="fvc-val" data-speed-val>4.0</span>
  </label>
  <button type="button" class="fvc-swap" data-swap>Show CNN</button>
</div>
`;

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

export function mountWidget(root: HTMLElement, weightsUrl = 'cnn.bin'): void {
  root.classList.add('fvc');
  root.innerHTML = MARKUP;
  const q = <T extends HTMLElement>(sel: string, scope: ParentNode = root) => scope.querySelector(sel) as T;

  const controls = {
    rmo: q<HTMLInputElement>('[data-rmo]'),
    threshold: q<HTMLInputElement>('[data-threshold]'),
    speed: q<HTMLInputElement>('[data-speed]'),
    seed: q<HTMLElement>('[data-seed]'),
  };
  const droneConfig = { ...DRONE_CONFIG };

  function createPanel(who: 'fly' | 'cnn', color: number): Panel {
    const section = q(`.fvc-panel[data-who="${who}"]`);
    const view = q<HTMLCanvasElement>('[data-view]', section);
    const { scene, droneMesh } = createWorld([], CORRIDOR.length, CORRIDOR.width);
    (droneMesh.material as THREE.MeshStandardMaterial).color.set(color);
    return {
      view,
      eye: q<HTMLCanvasElement>('[data-eye]', section).getContext('2d') as CanvasRenderingContext2D,
      trace: q<HTMLCanvasElement>('[data-trace]', section).getContext('2d') as CanvasRenderingContext2D,
      hits: q('[data-hits]', section),
      color: `#${color.toString(16).padStart(6, '0')}`,
      renderer: createRenderer(view),
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

  const panels = { fly: createPanel('fly', FLY_COLOR), cnn: createPanel('cnn', CNN_COLOR) };

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
      // Otherwise the shaders compile on the first frame, which stalls the page as the widget scrolls into view.
      // The drone camera draws into an sRGB target, which needs its own shader variants, so compile both.
      void panel.renderer.compileAsync(scene, panel.chase);
      panel.renderer.setRenderTarget(panel.droneCam.target);
      void panel.renderer.compileAsync(scene, panel.droneCam.camera);
      panel.renderer.setRenderTarget(null);
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

  // The render happens synchronously inside the call, so the drone only needs hiding around it, not until the
  // read-back lands.
  function capture(panel: Panel): Promise<Float32Array> {
    panel.droneMesh.visible = false;
    const pending = captureFrameAsync(panel.renderer, panel.scene, panel.droneCam, panel.drone.x, panel.drone.z);
    panel.droneMesh.visible = true;
    return pending;
  }

  function stepPanel(panel: Panel, frame: Float32Array, steerFrom: (frame: Float32Array) => number): void {
    drawEye(panel, frame);

    stepDrone(panel.drone, steerFrom(frame), DT, droneConfig);
    panel.drone.x = Math.max(-CORRIDOR.width / 2, Math.min(CORRIDOR.width / 2, panel.drone.x));

    const hit = checkCollision(panel.drone.x, panel.drone.z, obstacles);
    if (hit !== null && hit !== panel.lastHit) panel.hits.textContent = String(++panel.collisions);
    panel.lastHit = hit;
  }

  function render(panel: Panel): void {
    const { clientWidth, clientHeight } = panel.view;
    if (clientWidth === 0) return; // the panel swapped out on a phone: keep simulating, skip drawing
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

  async function frame(): Promise<void> {
    const { fly, cnn } = panels;
    const [flyFrame, cnnFrame] = await Promise.all([capture(fly), capture(cnn)]);

    let gfDrive = 0;
    stepPanel(fly, flyFrame, (f) => {
      const step = stepFlyController(flyController, f, DT);
      gfDrive = step.debug.gfDrive;
      return step.steering;
    });
    drawTrace(fly, gfDrive, Math.max(2.5, flyController.config.escapeThreshold * 1.5), 0, flyController.config.escapeThreshold);

    stepPanel(cnn, cnnFrame, (f) => (cnnController ? stepCnnController(cnnController, f, DT) : 0));
    drawTrace(cnn, cnnController ? cnnController.steering : 0, 1, -1, 0);

    render(fly);
    render(cnn);

    if (Math.min(fly.drone.z, cnn.drone.z) >= CORRIDOR.length) {
      seed++;
      loadCorridor();
    }
  }

  // On a long page this widget is usually off screen, and two WebGL contexts are not free: only run while visible.
  let onScreen = false;
  let running = false;
  async function tick(): Promise<void> {
    if (!onScreen || document.hidden) {
      running = false;
      return;
    }
    await frame();
    requestAnimationFrame(tick);
  }
  function resume(): void {
    if (running || !onScreen || document.hidden) return;
    running = true;
    requestAnimationFrame(tick);
  }
  new IntersectionObserver((entries) => {
    onScreen = entries[0].isIntersecting;
    resume();
  }).observe(root);
  document.addEventListener('visibilitychange', resume);

  q('[data-restart]').onclick = restart;
  q('[data-next-seed]').onclick = () => {
    seed++;
    loadCorridor();
  };
  controls.rmo.onchange = restart;
  controls.threshold.oninput = () => {
    q('[data-threshold-val]').textContent = Number(controls.threshold.value).toFixed(2);
    flyController.config.escapeThreshold = Number(controls.threshold.value);
  };
  controls.speed.oninput = () => {
    q('[data-speed-val]').textContent = Number(controls.speed.value).toFixed(1);
    droneConfig.speed = Number(controls.speed.value);
  };
  const swap = q('[data-swap]');
  const narrow = window.matchMedia('(max-width: 720px)');
  function swapPanels(): void {
    const flyPanel = q('.fvc-panel[data-who="fly"]');
    const cnnPanel = q('.fvc-panel[data-who="cnn"]');
    const showingFly = !flyPanel.classList.contains('is-hidden');
    flyPanel.classList.toggle('is-hidden', showingFly);
    cnnPanel.classList.toggle('is-hidden', !showingFly);
    swap.textContent = showingFly ? 'Show fly circuit' : 'Show CNN';
  }
  swap.onclick = swapPanels;
  // On a phone only one drone is on screen, and the simulation itself is the biggest thing to tap.
  for (const stage of root.querySelectorAll<HTMLElement>('.fvc-stage')) {
    stage.addEventListener('click', () => {
      if (narrow.matches) swapPanels();
    });
  }
  // The CSS drops to one column on narrow screens, so start on the fly panel.
  if (narrow.matches) q('.fvc-panel[data-who="cnn"]').classList.add('is-hidden');

  fetch(weightsUrl)
    .then((r) => r.arrayBuffer())
    .then((b) => {
      cnnController = createCnnController(loadCnnWeights(new Uint8Array(b)));
    });

  loadCorridor();
  Object.assign(root, { flyvscnn: { panels, frame } }); // handle for headless checks, scoped to the element
}

const style = document.createElement('style');
style.textContent = STYLE;
document.head.append(style);

for (const host of document.querySelectorAll<HTMLElement>('[data-fly-vs-cnn]')) {
  mountWidget(host, host.dataset.weights || 'cnn.bin');
}
