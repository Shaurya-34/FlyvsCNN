// Confirms the TypeScript forward pass matches PyTorch on the sample train/train.py saved.
import { readFileSync } from 'node:fs';
import { cnnForward, loadCnnWeights } from '../src/controllers/cnn';

const weights = loadCnnWeights(readFileSync('public/cnn.bin'));
const { input, output } = JSON.parse(readFileSync('data/cnn_check.json', 'utf8'));
const ts = cnnForward(weights, Float32Array.from(input));
const diff = Math.abs(ts - output);

console.log(`torch ${output.toFixed(6)}  ts ${ts.toFixed(6)}  diff ${diff.toExponential(2)}`);
if (diff > 1e-4) {
  console.error('MISMATCH: the TypeScript forward pass disagrees with PyTorch');
  process.exit(1);
}
