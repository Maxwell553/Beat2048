"""Seeded policy distillation with D4 augmentation and validation by whole game."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from bot2048.model import PolicyNet, encode, save_policy


# CCW board rotation maps U,R,D,L to L,U,R,D. Reflection swaps R/L.
def augment(boards, targets, rng):
    b = boards.copy()
    y = targets.copy()
    transforms = rng.integers(0, 8, len(b))
    for t in range(8):
        mask = transforms == t
        k = t % 4
        b[mask] = np.rot90(b[mask], k, axes=(1, 2))
        y[mask] = (y[mask] - k) % 4
        if t >= 4:
            b[mask] = b[mask, :, ::-1]
            y[mask] = (-y[mask]) % 4
    return b, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/teacher.npz")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="models/final_policy.pt")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.use_deterministic_algorithms(True)
    model = PolicyNet()
    save_policy(
        "models/initial_policy.pt",
        model,
        trained=False,
        seed=a.seed,
        description="Untrained randomly initialized neural policy; random baseline uses saved uniform action configuration.",
    )
    Path("models/random_agent.json").write_text(
        json.dumps(
            {
                "type": "uniform_legal_random",
                "policy_seed_rule": "environment_seed + 314159",
                "initial_policy": "initial_policy.pt",
                "trained": False,
            },
            indent=2,
        )
    )
    d = np.load(a.data)
    boards = d["boards"]
    targets = np.argmax(d["scores"], axis=1).astype(np.int64)
    ids = np.unique(d["episodes"])
    rng.shuffle(ids)
    val_ids = ids[: max(1, len(ids) // 5)]
    val = np.isin(d["episodes"], val_ids)
    train = np.flatnonzero(~val)
    vx = encode(boards[val])
    vy = torch.tensor(targets[val])
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, a.epochs, eta_min=0.00005
    )
    best = float("inf")
    history = []
    start = time.time()
    for epoch in range(a.epochs):
        model.train()
        rng.shuffle(train)
        total = correct = count = 0
        for offset in range(0, len(train), a.batch_size):
            indices = train[offset : offset + a.batch_size]
            b, y = augment(boards[indices], targets[indices], rng)
            x = encode(b)
            y = torch.tensor(y.copy())
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        model.eval()
        with torch.no_grad():
            logits = model(vx)
            vloss = nn.functional.cross_entropy(logits, vy).item()
            accuracy = (logits.argmax(1) == vy).float().mean().item()
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "train_accuracy": correct / count,
            "validation_loss": vloss,
            "validation_accuracy": accuracy,
            "seconds": time.time() - start,
        }
        history.append(row)
        if vloss < best:
            best = vloss
            save_policy(
                a.output,
                model,
                trained=True,
                seed=a.seed,
                epoch=epoch + 1,
                validation_loss=vloss,
                validation_accuracy=accuracy,
                training_states=len(train),
                validation_states=int(val.sum()),
                validation_episode_ids=val_ids.tolist(),
                parameter_count=sum(p.numel() for p in model.parameters()),
            )
        print(json.dumps(row), flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/training.json").write_text(
        json.dumps(
            {
                "config": vars(a),
                "history": history,
                "training_episode_ids": ids[len(val_ids) :].tolist(),
                "validation_episode_ids": val_ids.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
