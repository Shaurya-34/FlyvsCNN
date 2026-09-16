// Usage: npx tsx scripts/gen-episodes.ts [trainEpisodes=200] [--three]
// Writes data/frames.npy (N,48,64 uint8), data/steering.npy (N, float32), data/episode.npy (N, int32 seed).
// Val episodes use VAL_SEEDS (seed >= 5001), so split by seed, not by frame. --three renders with real Three.js.
import { mkdirSync, writeFileSync } from 'node:fs';
import { TRAIN_SEEDS, VAL_SEEDS } from '../src/sim/seeds';
import { H, W } from './corridor-harness';
import { args, openRunner, useThree } from './runner';

const nTrain = Number(args[0] ?? 200);
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

const runner = await openRunner();
for (const [i, seed] of seeds.entries()) {
  const r = await runner.record(seed);
  frames.push(r.frames);
  for (const s of r.steering) {
    steering.push(s);
    episode.push(seed);
  }
  collisions += r.collisions;
  if ((i + 1) % 20 === 0) console.log(`${i + 1}/${seeds.length} episodes`);
}
await runner.close();

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
writeFileSync('data/frames.npy', npy('|u1', [steering.length, H, W], Buffer.concat(frames)));
writeFileSync('data/steering.npy', npy('<f4', [steering.length], bytes(Float32Array.from(steering))));
writeFileSync('data/episode.npy', npy('<i4', [episode.length], bytes(Int32Array.from(episode))));

console.log(
  `${seeds.length} episodes (${nTrain} train, ${nVal} val) on the ${useThree ? 'Three.js' : 'harness'} renderer, ` +
    `${steering.length} frames, ${collisions} collisions while recording`,
);
