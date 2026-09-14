import { generateCorridor } from '../sim/corridor';
import { createDrone, stepDrone } from '../sim/drone';
import { checkCollision } from '../sim/collisions';
import { createScene, buildCorridorMeshes } from '../render/scene';
import { createDroneCamera, captureFrame, horizontalFovDeg } from '../render/droneCamera';
import { createFlyController, stepFlyController } from '../controllers/flyCircuit';

const SEED = 42;
const CORRIDOR_LENGTH = 200;
const CORRIDOR_WIDTH = 8;
const DENSITY = 0.3;
const SIZE_VARIANCE = 0.4;
const DRONE_CONFIG = { speed: 4, lateralDamping: 2 };

const canvas = document.getElementById('viewport') as HTMLCanvasElement;
const { scene, camera, renderer, droneMesh } = createScene(canvas);

const obstacles = generateCorridor({
  seed: SEED,
  length: CORRIDOR_LENGTH,
  width: CORRIDOR_WIDTH,
  density: DENSITY,
  sizeVariance: SIZE_VARIANCE,
});
buildCorridorMeshes(scene, obstacles, CORRIDOR_LENGTH, CORRIDOR_WIDTH);

const drone = createDrone();

const droneCam = createDroneCamera();
const flyController = createFlyController({
  hFovDeg: horizontalFovDeg(droneCam.camera),
  vFovDeg: droneCam.camera.fov,
});

function resize(): void {
  const { clientWidth, clientHeight } = canvas;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let lastCollisionIndex: number | null = null;
let collisionCount = 0;
let lastTime = performance.now();

function tick(now: number): void {
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  droneMesh.visible = false;
  const frame = captureFrame(renderer, scene, droneCam, drone.x, drone.z);
  droneMesh.visible = true;

  const { steering } = stepFlyController(flyController, frame, dt);
  stepDrone(drone, steering, dt, DRONE_CONFIG);

  const hitIndex = checkCollision(drone.x, drone.z, obstacles);
  if (hitIndex !== null && hitIndex !== lastCollisionIndex) {
    collisionCount++;
    console.log(`collision #${collisionCount} with obstacle ${hitIndex} at z=${drone.z.toFixed(1)}`);
  }
  lastCollisionIndex = hitIndex;

  droneMesh.position.set(drone.x, 0.6, drone.z);
  camera.position.set(drone.x, 2, drone.z - 6);
  camera.lookAt(drone.x, 0.6, drone.z + 4);

  renderer.render(scene, camera);

  if (drone.z < CORRIDOR_LENGTH) {
    requestAnimationFrame(tick);
  }
}
requestAnimationFrame(tick);
