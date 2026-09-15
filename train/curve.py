# Data-efficiency curve: train the CNN on growing numbers of expert episodes with the same gradient-step budget,
# benchmark each in closed loop on the same corridors as the fly circuit, and plot.
# Usage, from the repo root: python train/curve.py   (re-running skips models that already exist)
import json
import math
import os
import subprocess
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPISODES = [2, 5, 20, 50, 200]
BATCH = 256
STEPS = 9360  # 20 epochs over 200 episodes at batch 256, the default full training run

episode = np.load("data/episode.npy")
os.makedirs("train/curve", exist_ok=True)

paths = []
for n in EPISODES:
    path = f"train/curve/cnn_{n}.bin"
    if not os.path.exists(path):
        samples = int((episode <= n).sum()) - 3 * n  # first 3 frames of each run lack a full 4-frame stack
        epochs = math.ceil(STEPS / math.ceil(samples / BATCH))
        print(f"\n=== {n} episodes: {samples} samples, {epochs} epochs ===", flush=True)
        subprocess.run([sys.executable, "train/train.py", str(epochs), str(n), path], check=True)
    paths.append(path)

subprocess.run("npx tsx scripts/verify-fly-controller.ts " + " ".join(paths), shell=True, check=True)
rows = {r["label"]: r for r in json.load(open("data/benchmark.json"))}

cnn = [rows[p]["collisions"] for p in paths]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(EPISODES, cnn, "o-", color="#d62728", label="CNN")
lines = [
    ("fly circuit, untuned (0 episodes)", "fly, RMO on", "#1f77b4", "--"),
    ("fly circuit, tuned on 20 training corridors", "fly, tuned", "#1f77b4", ":"),
    ("no controller", "no controller", "0.6", "--"),
    ("expert (sees obstacles)", "expert (cheat)", "0.2", "--"),
]
for label, key, color, style in lines:
    if key in rows:
        ax.axhline(rows[key]["collisions"], color=color, linestyle=style, linewidth=1, label=label)
ax.set_xscale("log")
ax.set_xticks(EPISODES, [str(n) for n in EPISODES])
ax.set_xlabel("CNN training episodes (expert demonstrations)")
ax.set_ylabel("collisions, 20 benchmark corridors")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("train/curve/data_efficiency.png", dpi=150)

print("\nepisodes  collisions")
for n, c in zip(EPISODES, cnn):
    print(f"{n:8d}  {c:10d}")
print(f"fly circuit: {rows['fly, RMO on']['collisions']}")
