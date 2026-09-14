import * as THREE from 'three';

const FRAME_WIDTH = 64;
const FRAME_HEIGHT = 48;
const DRONE_CAM_FOV_DEG = 90; // vertical FOV, an engineering/sensor choice, not biology

export function createDroneCamera() {
  const camera = new THREE.PerspectiveCamera(DRONE_CAM_FOV_DEG, FRAME_WIDTH / FRAME_HEIGHT, 0.05, 200);
  const target = new THREE.WebGLRenderTarget(FRAME_WIDTH, FRAME_HEIGHT);
  const pixelBuffer = new Uint8Array(FRAME_WIDTH * FRAME_HEIGHT * 4);
  return { camera, target, pixelBuffer };
}

export function horizontalFovDeg(camera: THREE.PerspectiveCamera): number {
  const vFovRad = (camera.fov * Math.PI) / 180;
  const hFovRad = 2 * Math.atan(Math.tan(vFovRad / 2) * camera.aspect);
  return (hFovRad * 180) / Math.PI;
}

// The drone's own forward view as grayscale: the ONLY thing any controller sees. No scene graph, just pixels.
export function captureFrame(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  handles: ReturnType<typeof createDroneCamera>,
  droneX: number,
  droneZ: number,
): Float32Array {
  const { camera, target, pixelBuffer } = handles;
  camera.position.set(droneX, 0.6, droneZ);
  camera.lookAt(droneX, 0.6, droneZ + 1);

  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.readRenderTargetPixels(target, 0, 0, FRAME_WIDTH, FRAME_HEIGHT, pixelBuffer);
  renderer.setRenderTarget(null);

  const gray = new Float32Array(FRAME_WIDTH * FRAME_HEIGHT);
  for (let i = 0; i < FRAME_WIDTH * FRAME_HEIGHT; i++) {
    const r = pixelBuffer[i * 4];
    const g = pixelBuffer[i * 4 + 1];
    const b = pixelBuffer[i * 4 + 2];
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  }
  return gray;
}
