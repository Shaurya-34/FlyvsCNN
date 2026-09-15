// Hand-written forward pass for train/model.py's RegressionCNN, using weights exported by train/train.py, so the browser
// needs no ML runtime. Same contract as every controller: 64x48 grayscale frames in, steering out.

const FRAME = 64 * 48;
const FRAME_INTERVAL_S = 0.05; // training frames were recorded at 20Hz

type Layer = { w: Float32Array; b: Float32Array };

// Order must match model.parameters() in PyTorch.
export function loadCnnWeights(bytes: Uint8Array) {
  const all = new Float32Array(new Uint8Array(bytes).buffer);
  let offset = 0;
  const take = (n: number) => all.subarray(offset, (offset += n));
  const weights = {
    conv1: { w: take(16 * 4 * 25), b: take(16) },
    conv2: { w: take(32 * 16 * 9), b: take(32) },
    conv3: { w: take(16 * 32 * 9), b: take(16) },
    fc1: { w: take(32 * 16 * 6 * 8), b: take(32) },
    fc2: { w: take(32), b: take(1) },
  };
  if (offset !== all.length) throw new Error(`cnn weights: expected ${offset} floats, got ${all.length}`);
  return weights;
}

export type CnnWeights = ReturnType<typeof loadCnnWeights>;

// Stride-2 conv with "same"-style padding, so each layer halves height and width. ReLU applied.
function conv(x: Float32Array, inC: number, inH: number, inW: number, layer: Layer, k: number): Float32Array {
  const outC = layer.b.length;
  const outH = inH / 2;
  const outW = inW / 2;
  const pad = (k - 1) / 2;
  const out = new Float32Array(outC * outH * outW);

  for (let oc = 0; oc < outC; oc++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let sum = layer.b[oc];
        for (let ic = 0; ic < inC; ic++) {
          for (let ky = 0; ky < k; ky++) {
            const iy = oy * 2 - pad + ky;
            if (iy < 0 || iy >= inH) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = ox * 2 - pad + kx;
              if (ix < 0 || ix >= inW) continue;
              sum += x[(ic * inH + iy) * inW + ix] * layer.w[((oc * inC + ic) * k + ky) * k + kx];
            }
          }
        }
        out[(oc * outH + oy) * outW + ox] = Math.max(0, sum);
      }
    }
  }
  return out;
}

function dense(x: Float32Array, layer: Layer): Float32Array {
  const out = new Float32Array(layer.b.length);
  for (let o = 0; o < out.length; o++) {
    let sum = layer.b[o];
    for (let i = 0; i < x.length; i++) sum += layer.w[o * x.length + i] * x[i];
    out[o] = sum;
  }
  return out;
}

// stack: 4 frames, oldest first, 4x48x64 channel-first.
export function cnnForward(w: CnnWeights, stack: Float32Array): number {
  const c1 = conv(stack, 4, 48, 64, w.conv1, 5);
  const c2 = conv(c1, 16, 24, 32, w.conv2, 3);
  const c3 = conv(c2, 32, 12, 16, w.conv3, 3);
  const hidden = dense(c3, w.fc1).map((v) => Math.max(0, v));
  return Math.tanh(dense(hidden, w.fc2)[0]);
}

export function createCnnController(weights: CnnWeights) {
  return { weights, frames: [] as Float32Array[], t: 0, nextFrameT: 0, steering: 0 };
}

export type CnnController = ReturnType<typeof createCnnController>;

// Frames enter the stack at 20Hz, matching the training data. Steering is held between frames.
export function stepCnnController(c: CnnController, frame: Float32Array, dt: number): number {
  const due = c.t >= c.nextFrameT - 1e-6;
  c.t += dt;
  if (!due) return c.steering;

  c.nextFrameT += FRAME_INTERVAL_S;
  c.frames.push(Float32Array.from(frame, (v) => Math.round(v * 255) / 255)); // same uint8 quantization as the dataset
  if (c.frames.length > 4) c.frames.shift();
  if (c.frames.length === 4) {
    const stack = new Float32Array(4 * FRAME);
    c.frames.forEach((f, i) => stack.set(f, i * FRAME));
    c.steering = cnnForward(c.weights, stack);
  }
  return c.steering;
}
