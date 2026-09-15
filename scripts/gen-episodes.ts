// Usage: npx tsx scripts/gen-episodes.ts [trainEpisodes=200]
// Writes data/frames.npy (N,48,64 uint8), data/steering.npy (N, float32), data/episode.npy (N, int32 seed).
// Val episodes use VAL_SEEDS (seed >= 5001), so split by seed, not by frame.
import { mkdirSync, writeFileSync } from 'node:fs';
import { mulberry32 } from '../src/sim/corridor';
import { TRAIN_SEEDS, VAL_SEEDS } from '../src/sim/seeds';
import { flyEpisode, W, H } from './corridor-harness';
import { expertSteer } from './expert';

const RECORD_EVERY = 3; // 60Hz sim -> 20Hz frames. The CNN's 4-frame stack must use this same spacing at inference.

const nTrain = Number(process.argv[2] ?? 200);
const nVal = Math.min(Math.ceil(nTrain / 10), VAL_SEEDS.to - VAL_SEEDS.from + 1);
const seeds = [
  ...Array.from({ length: nTrain }, (_, i) => TRAIN_SEEDS.from + i),
  ...Array.from({ length: nVal }, (_, i) => VAL_SEEDS.from + i),
];

// ponytail: whole dataset held in memory as one file, fine to ~1000 episodes (~2GB); shard when scaling past that.
const frames: Uint8Array[] = [];
const steering: number[] = [];
const episode: number[] = [];
let collisions = 0;

for (const seed of seeds) {
  const rng = mulberry32(seed + 1_000_000); // separate stream from the corridor layout
  let noise = 0;
  let step = 0;

  collisions += flyEpisode(seed, (frame, drone, obstacles) => {
    const label = expertSteer(drone, obstacles);
    if (step % RECORD_EVERY === 0) {
      frames.push(Uint8Array.from(frame, (v) => Math.round(v * 255)));
      steering.push(label);
      episode.push(seed);
    }
    // Occasional steering noise so the data includes recovering from bad positions. Labels stay clean.
    if (step % 30 === 0) noise = rng() < 0.3 ? (rng() - 0.5) * 1.2 : 0;
    step++;
    return Math.max(-1, Math.min(1, label + noise));
  }).collisions;
}

function npy(descr: string, shape: number[], data: Uint8Array): Buffer {
  const dims = shape.join(', ') + (shape.length === 1 ? ',' : '');
  const dict = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${dims}), }`;
  const header = dict.padEnd(Math.ceil((dict.length + 11) / 64) * 64 - 11) + '\n';
  const prefix = Buffer.alloc(10);
  prefix.write('\x93NUMPY', 0, 'latin1');
  prefix[6] = 1;
  prefix.writeUInt16LE(header.length, 8);
  return Buffer.concat([prefix, Buffer.from(header, 'latin1'), data]);
}

const bytes = (a: Float32Array | Int32Array) => new Uint8Array(a.buffer);

mkdirSync('data', { recursive: true });
writeFileSync('data/frames.npy', npy('|u1', [frames.length, H, W], Buffer.concat(frames)));
writeFileSync('data/steering.npy', npy('<f4', [steering.length], bytes(Float32Array.from(steering))));
writeFileSync('data/episode.npy', npy('<i4', [episode.length], bytes(Int32Array.from(episode))));

console.log(`${seeds.length} episodes (${nTrain} train, ${nVal} val), ${frames.length} frames, ${collisions} collisions while recording`);
