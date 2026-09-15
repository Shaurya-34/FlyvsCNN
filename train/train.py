# Usage, from the repo root: python train/train.py [epochs=20] [train_episodes=all] [out=public/cnn.bin]
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import RegressionCNN

DEFAULT_OUT = "public/cnn.bin"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TRAIN_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else None  # first N train seeds; None = all
OUT = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
PATIENCE_STEPS = 1000  # stop once validation loss hasn't improved for this many optimizer steps
VAL_SEED_START = 5001  # VAL_SEEDS in src/sim/seeds.ts

torch.manual_seed(0)


class CorridorDataset(Dataset):
    def __init__(self, split, episodes=None):
        self.frames = np.load("data/frames.npy", mmap_mode="r")
        self.steering = np.load("data/steering.npy")
        episode = np.load("data/episode.npy")

        # index i needs frames i-3..i, all from the same run (runs are stored back to back)
        full_stack = np.zeros(len(episode), dtype=bool)
        full_stack[3:] = episode[3:] == episode[:-3]

        is_val = episode >= VAL_SEED_START
        wanted = is_val if split == "val" else ~is_val
        if episodes is not None:
            wanted &= episode <= episodes
        self.idx = np.where(full_stack & wanted)[0]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        x = torch.from_numpy(self.frames[i - 3 : i + 1].astype(np.float32) / 255)  # (4, 48, 64)
        return x, torch.tensor(self.steering[i])


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            loss = criterion(model(x), y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
    return total / len(loader.dataset)


device = "cuda" if torch.cuda.is_available() else "cpu"
train_set, val_set = CorridorDataset("train", TRAIN_EPISODES), CorridorDataset("val")
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024)

model = RegressionCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"{len(train_set)} train / {len(val_set)} val samples, {sum(p.numel() for p in model.parameters())} params, {device}")
print(f"baseline val loss (always steer 0): {float(np.mean(val_set.steering[val_set.idx] ** 2)):.4f}")

# Validate every epoch and stop early: small datasets overfit within a few epochs.
best, best_state, steps_since_best = float("inf"), None, 0
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, optimizer)
    val_loss = run_epoch(model, val_loader)
    steps_since_best += len(train_loader)
    marker = ""
    if val_loss < best:
        best, steps_since_best = val_loss, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        marker = "  best"
    if marker or epoch % max(1, EPOCHS // 40) == 0:
        print(f"epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}{marker}", flush=True)
    if steps_since_best >= PATIENCE_STEPS:
        print(f"early stop at epoch {epoch}", flush=True)
        break

# Export the best checkpoint as flat float32 for the TypeScript forward pass (src/controllers/cnn.ts).
model.load_state_dict(best_state)
model.eval().cpu()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.concatenate([p.detach().numpy().ravel() for p in model.parameters()]).astype("<f4").tofile(OUT)

# One input/output pair so scripts/check-cnn.ts can confirm the TypeScript and PyTorch forward passes agree.
if OUT == DEFAULT_OUT:
    x, _ = val_set[0]
    with torch.no_grad():
        output = model(x.unsqueeze(0)).item()
    with open("data/cnn_check.json", "w") as f:
        json.dump({"input": x.numpy().ravel().tolist(), "output": output}, f)

print(f"exported {OUT} (best val {best:.4f})")
