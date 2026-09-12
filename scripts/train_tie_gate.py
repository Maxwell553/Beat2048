"""Train a small neural selector for two direct neural tie-break actions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TieGate(nn.Module):
    def __init__(self, inputs, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--regret-weight", type=float, default=1.0)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=2048)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = np.load(args.train)
    validation = np.load(args.validation)
    model = TieGate(train["features"].shape[1], args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    order = np.arange(len(train["labels"]))
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        model.train()
        total = 0
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset : offset + args.batch_size]
            x = torch.as_tensor(train["features"][ids], device=args.device)
            y = torch.as_tensor(train["labels"][ids], device=args.device)
            weights = 1 + args.regret_weight * torch.log1p(
                torch.as_tensor(train["regrets"][ids], device=args.device) / 1000
            )
            per = nn.functional.binary_cross_entropy_with_logits(
                model(x), y, reduction="none"
            )
            loss = (per * weights).sum() / weights.sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            vx = torch.as_tensor(validation["features"], device=args.device)
            vy = torch.as_tensor(validation["labels"], device=args.device)
            logits = model(vx)
            validation_loss = nn.functional.binary_cross_entropy_with_logits(logits, vy).item()
            accuracy = ((logits >= 0) == vy.bool()).float().mean().item()
        row = {"epoch": epoch, "train_loss": total / len(order), "validation_loss": validation_loss, "validation_accuracy": accuracy}
        if validation_loss < best:
            best = validation_loss
            artifact = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "metadata": {"inputs": train["features"].shape[1], "hidden": args.hidden, "config": vars(args), **row},
            }
            torch.save(artifact, str(args.output) + ".tmp")
            Path(str(args.output) + ".tmp").replace(args.output)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
