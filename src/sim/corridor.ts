// Seeded PRNG so a given seed always reproduces the same corridor.
function mulberry32(seed: number): () => number {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type ObstacleType = 'pole' | 'box';

export interface Obstacle {
  type: ObstacleType;
  x: number; // lateral position
  z: number; // forward distance down the corridor
  height: number;
  radius?: number; // poles
  width?: number; // boxes
  depth?: number; // boxes
}

export interface CorridorConfig {
  seed: number;
  length: number; // corridor length along z
  width: number; // corridor width along x
  density: number; // obstacles per unit length
  sizeVariance: number; // 0..1 spread applied to base sizes
}

// Irregular spacing and lateral placement on purpose: a rhythmic layout would
// let the CNN learn a spacing pattern and dodge blind, which would invalidate
// the fly-vs-CNN comparison.
export function generateCorridor(config: CorridorConfig): Obstacle[] {
  const { seed, length, width, density, sizeVariance } = config;
  const rng = mulberry32(seed);
  const obstacles: Obstacle[] = [];

  const baseSpacing = 1 / density;
  let z = baseSpacing * (0.5 + rng());

  while (z < length) {
    const type: ObstacleType = rng() < 0.5 ? 'pole' : 'box';
    const x = (rng() - 0.5) * width * 0.9;
    const sizeJitter = 1 + (rng() - 0.5) * 2 * sizeVariance;

    if (type === 'pole') {
      obstacles.push({
        type,
        x,
        z,
        radius: 0.4 * sizeJitter,
        height: 3 * sizeJitter,
      });
    } else {
      obstacles.push({
        type,
        x,
        z,
        width: 0.8 * sizeJitter,
        depth: 0.8 * sizeJitter,
        height: 2 * sizeJitter,
      });
    }

    z += baseSpacing * (0.4 + rng() * 1.6);
  }

  return obstacles;
}
