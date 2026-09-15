export const TRAIN_SEEDS = { from: 1, to: 2000 };
export const VAL_SEEDS = { from: 5001, to: 5100 }; // offline validation loss and checkpoint selection
export const BENCH_SEEDS = { from: 5101, to: 5200 }; // closed-loop benchmark while developing, no overlap with VAL_SEEDS
export const EVAL_SEEDS = { from: 9000, to: 9099 }; // held out: Phase 7 evaluation only, never train or tune on these
