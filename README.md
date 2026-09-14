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

## Status

- [x] Phase 1: seeded corridor, drone kinematics, Three.js renderer
- [x] Phase 2: circuit verified on a synthetic looming disc (`prototypes/fly_circuit.py`)
- [x] Phase 3: circuit on rendered frames, with RMO
- [ ] Phase 4: headless episode generation + CNN training
- [ ] Phase 5: parameter sweeps
- [ ] Phase 6: widget UI
- [ ] Phase 7: evaluation on held-out seeds

Phase 3 results, 8 corridors:

| condition | collisions | escape frames | avg abs steering |
|---|---|---|---|
| no controller | 49 | 9673 | 0.755 |
| fly circuit, RMO on | 13 | 5551 | 0.517 |
| fly circuit, RMO off | 11 | 12043 | 0.876 |

With RMO off the drone is in escape mode most of the time and avoids obstacles mainly by swerving constantly.

## Layout

```
src/sim          corridor generation, drone kinematics, collisions (no DOM)
src/controllers  flyCircuit.ts: frame in, steering out
src/render       Three.js scene and the drone's 64x48 camera
src/widget       browser entry point
scripts          headless corridor harness and controller verification
prototypes       Python prototype of the circuit
```

## Run

```bash
npm install
npm run dev                                   # browser demo
npx tsx scripts/verify-fly-controller.ts      # headless collision benchmark
python prototypes/fly_circuit.py              # synthetic looming-disc check
```
