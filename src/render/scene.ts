import * as THREE from 'three';
import type { Obstacle } from '../sim/corridor';

export function createRenderer(canvas: HTMLCanvasElement): THREE.WebGLRenderer {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  return renderer;
}

// Everything the drone can see. Shared by the widget and the headless bench page, so both render identically.
export function createWorld(obstacles: Obstacle[], corridorLength: number, corridorWidth: number) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf2f2f0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  const directional = new THREE.DirectionalLight(0xffffff, 0.6);
  directional.position.set(5, 10, 5);
  scene.add(ambient, directional);

  const droneMesh = new THREE.Mesh(
    new THREE.ConeGeometry(0.3, 0.6, 8),
    new THREE.MeshStandardMaterial({ color: 0x333333, flatShading: true }),
  );
  droneMesh.rotation.x = Math.PI / 2;
  scene.add(droneMesh);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(corridorWidth * 3, corridorLength),
    new THREE.MeshStandardMaterial({ color: 0xdedede }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.z = corridorLength / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(Math.max(corridorWidth * 3, corridorLength), 40, 0xbbbbbb, 0xd8d8d8);
  grid.position.set(0, 0.01, corridorLength / 2);
  scene.add(grid);

  const obstacleMat = new THREE.MeshStandardMaterial({ color: 0x9a9a9a, flatShading: true });
  for (const o of obstacles) {
    const mesh =
      o.type === 'pole'
        ? new THREE.Mesh(new THREE.CylinderGeometry(o.radius, o.radius, o.height, 12), obstacleMat)
        : new THREE.Mesh(new THREE.BoxGeometry(o.width, o.height, o.depth), obstacleMat);
    mesh.position.set(o.x, o.height / 2, o.z);
    scene.add(mesh);
  }

  const wallGeo = new THREE.PlaneGeometry(corridorLength, 4);
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xe8e8e6,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.15,
  });

  const leftWall = new THREE.Mesh(wallGeo, wallMat);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-corridorWidth / 2, 2, corridorLength / 2);
  scene.add(leftWall);

  const rightWall = leftWall.clone();
  rightWall.position.x = corridorWidth / 2;
  scene.add(rightWall);

  return { scene, droneMesh };
}
