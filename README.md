# Fly Circuit vs CNN

Two simulated drones fly the same procedurally generated obstacle corridor:

- **Fly circuit**: a hardwired model of the fruit fly's looming-escape pathway (LPLC2 + LC4 feeding the Giant Fiber). Zero training.
- **CNN**: a small conv net trained end-to-end on crash / no-crash episodes.

Both see only a downsampled 64x48 grayscale camera frame. Neither has access to obstacle positions or any other scene state. The question: can a circuit with a handful of biologically fixed constants and no training data match a learned network at the one job it evolved for?

Built as an interactive demo for a post on [shauryasharma.tech](https://shauryasharma.tech).

## The model

The circuit follows the static model in Ache et al. 2019, *Current Biology* 29:1073-1081: GF drive is a weighted sum of a log-Gaussian size term (LPLC2, peak at 42 deg) and a linear angular-velocity term (LC4). The paper's two inhibitory terms are dropped for now. The escape threshold and steering on top are our own additions.

Radial motion opponency (RMO) rejects expansion caused by the drone's own motion. It can be toggled off to show the detector panicking at self-motion.

Caveats: the connectome gives the wiring, not the transfer functions (those come from separate physiology). The published model was fit to a single expanding disc in a lab, so extending it to continuous corridor flight is a modeling choice, not established science.

The CNN (`train/model.py`) takes the last 4 frames stacked (20Hz), has 3 stride-2 convolutions and 2 fully connected layers (~35k parameters), and regresses steering. It is trained by imitation on labels from a scripted expert that is allowed to see obstacle positions; the trained CNN itself sees only frames.

## Status

- [x] Phase 1: seeded corridor, drone kinematics, Three.js renderer
- [x] Phase 2: circuit verified on a synthetic looming disc (`prototypes/fly_circuit.py`)
- [x] Phase 3: circuit on rendered frames, with RMO
- [ ] Phase 4: headless episode generation + CNN training
- [ ] Phase 5: parameter sweeps
- [ ] Phase 6: widget UI
- [ ] Phase 7: evaluation on held-out seeds

Benchmark, 20 corridors (seeds 5101-5120):

| condition | collisions | escape frames | avg abs steering |
|---|---|---|---|
| no controller | 118 | 24139 | 0.751 |
| fly circuit, RMO on | 43 | 14425 | 0.527 |
| fly circuit, RMO off | 28 | 30453 | 0.884 |
| expert (sees obstacles, not a contestant) | 6 | - | 0.147 |

With RMO off the drone is in escape mode most of the time and avoids obstacles mainly by swerving constantly.

Data efficiency (`python train/curve.py`): the CNN trained on N expert episodes (30s of flying each), same step budget and early stopping for every N, one training run per point:

| training episodes | 2 | 5 | 20 | 50 | 200 |
|---|---|---|---|---|---|
| CNN collisions | 106 | 41 | 11 | 14 | 11 |

The fly circuit (43, zero episodes) is matched at about 5 episodes, roughly 2.5 minutes of demonstrations. Differences below ~5 collisions are within noise at 20 corridors.

Tuning the fly circuit (`npx tsx scripts/tune-fly.ts`): random search over its 7 engineering constants on the same 20 training corridors, published biology constants fixed, scored as collisions + 50 x average steering effort (scoring collisions alone found settings that swerve constantly). On the benchmark the tuned circuit gets 44 collisions at 0.36 average steering, versus 43 at 0.53 untuned and 11 at 0.15 for the CNN trained on those corridors. Tuning makes the circuit calmer but no safer.

![data efficiency](train/curve/data_efficiency.png)

Seeds (`src/sim/seeds.ts`): train 1-2000, validation 5001-5100 (checkpoint selection), benchmark 5101-5200, held-out evaluation 9000-9099. Nothing trains or tunes on the held-out range.

## Layout

```
src/sim          corridor generation, drone kinematics, collisions, seed split (no DOM)
src/controllers  flyCircuit.ts and cnn.ts: frames in, steering out
src/render       Three.js scene and the drone's 64x48 camera
src/widget       browser entry point
scripts          headless corridor harness, expert, data generation, benchmark
train            CNN model and training (PyTorch)
prototypes       Python prototype of the circuit
```

## Run

```bash
npm install
npm run dev                                   # browser demo
npx tsx scripts/verify-fly-controller.ts      # headless collision benchmark
python prototypes/fly_circuit.py              # synthetic looming-disc check
```

Train the CNN:

```bash
npx tsx scripts/gen-episodes.ts 200           # data/*.npy from the headless harness
python train/train.py 20                      # trains, exports public/cnn.bin
npx tsx scripts/check-cnn.ts                  # TypeScript forward pass matches PyTorch
npx tsx scripts/verify-fly-controller.ts      # benchmark now includes the CNN
```
