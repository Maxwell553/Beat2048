"""Train direct action logits from precomputed current-board neural features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import FusionPolicy


def features(data, ids, device):
    boards = torch.as_tensor(data["boards"][ids], device=device, dtype=torch.long)
    onehot = torch.nn.functional.one_hot(boards.clamp(0, 15), 16).flatten(1).float()
    rest = np.concatenate(
        [
            data["aligned"][ids].reshape(len(ids), -1),
            data["votes"][ids] / 8.0,
            data["probabilities"][ids],
            data["sparse"][ids].reshape(len(ids), -1),
            *( [data["experts"][ids].reshape(len(ids), -1)] if "experts" in data else [] ),
        ], axis=1,
    ).astype(np.float32)
    return torch.cat([onehot, torch.as_tensor(rest, device=device)], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument(
        "--train-targets",
        help="Optional .npy teacher scores aligned with the training archive",
    )
    p.add_argument(
        "--validation-targets",
        help="Optional .npy teacher scores aligned with the validation archive",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Initialize the fusion head from this checkpoint")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--regret-weight", type=float, default=0.0)
    p.add_argument("--late-state-weight", type=float, default=0.0)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=2048)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.train) as archive:
        train = {name: archive[name] for name in archive.files}
    with np.load(args.validation) as archive:
        validation = {name: archive[name] for name in archive.files}
    if args.train_targets:
        train["targets"] = np.load(args.train_targets)
    if args.validation_targets:
        validation["targets"] = np.load(args.validation_targets)
    if len(train["targets"]) != len(train["boards"]):
        raise ValueError("training targets are not aligned with the feature archive")
    if len(validation["targets"]) != len(validation["boards"]):
        raise ValueError("validation targets are not aligned with the feature archive")
    inputs = 256 + 32 + 4 + 4 + train["sparse"].shape[1] * 4
    if "experts" in train:
        inputs += int(np.prod(train["experts"].shape[1:]))
    model = FusionPolicy(inputs, args.hidden).to(args.device)
    if args.resume:
        resumed = torch.load(args.resume, map_location=args.device, weights_only=True)
        if resumed["metadata"]["inputs"] != inputs:
            raise ValueError("resume checkpoint input size does not match training data")
        model.load_state_dict(resumed["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    order = np.arange(len(train["boards"])); best = 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order); model.train(); total = correct = 0
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset : offset + args.batch_size]
            target = train["targets"][ids]
            legal = torch.as_tensor(target > -1e20, device=args.device)
            labels = torch.as_tensor(target.argmax(1), device=args.device)
            logits = model(features(train, ids, args.device)).masked_fill(~legal, -1e9)
            per_state = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            if args.regret_weight:
                q = torch.as_tensor(target, device=args.device)
                regret = (q.max(1, keepdim=True).values - q).clamp(0, 50000) / 1000
                regret = regret.masked_fill(~legal, 0)
                per_state = per_state + args.regret_weight * (
                    torch.softmax(logits, 1) * regret
                ).sum(1)
            if args.late_state_weight:
                largest = torch.as_tensor(
                    train["boards"][ids].max(axis=(1, 2)),
                    device=args.device, dtype=torch.float32,
                )
                weights = 1 + args.late_state_weight * (
                    (largest - 8).clamp(0, 2) / 2
                )
                loss = (per_state * weights).sum() / weights.sum()
            else:
                loss = per_state.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(ids); correct += int((logits.argmax(1) == labels).sum())
        model.eval(); validation_correct = validation_total = 0; validation_loss = 0
        with torch.no_grad():
            for offset in range(0, len(validation["boards"]), args.batch_size):
                ids = np.arange(offset, min(offset + args.batch_size, len(validation["boards"])))
                target = validation["targets"][ids]
                legal = torch.as_tensor(target > -1e20, device=args.device)
                labels = torch.as_tensor(target.argmax(1), device=args.device)
                logits = model(features(validation, ids, args.device)).masked_fill(~legal, -1e9)
                per_state = torch.nn.functional.cross_entropy(
                    logits, labels, reduction="none"
                )
                if args.regret_weight:
                    q = torch.as_tensor(target, device=args.device)
                    regret = (q.max(1, keepdim=True).values - q).clamp(0, 50000) / 1000
                    regret = regret.masked_fill(~legal, 0)
                    per_state = per_state + args.regret_weight * (
                        torch.softmax(logits, 1) * regret
                    ).sum(1)
                if args.late_state_weight:
                    largest = torch.as_tensor(
                        validation["boards"][ids].max(axis=(1, 2)),
                        device=args.device, dtype=torch.float32,
                    )
                    weights = 1 + args.late_state_weight * (
                        (largest - 8).clamp(0, 2) / 2
                    )
                    loss = (per_state * weights).sum() / weights.sum()
                else:
                    loss = per_state.mean()
                validation_loss += loss.item() * len(ids); validation_correct += int((logits.argmax(1) == labels).sum()); validation_total += len(ids)
        row = {"epoch": epoch, "train_loss": total / len(order), "train_accuracy": correct / len(order), "validation_loss": validation_loss / validation_total, "validation_accuracy": validation_correct / validation_total}
        artifact = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "metadata": {"inputs": inputs, "hidden": args.hidden, "config": vars(args), **row}}
        if row["validation_accuracy"] > best:
            best = row["validation_accuracy"]; torch.save(artifact, str(args.output) + ".tmp"); Path(str(args.output) + ".tmp").replace(args.output)
        if epoch in (1, 3, 6, 9, 12):
            destination = Path(args.output).with_name(f"{Path(args.output).stem}.epoch{epoch}.pt"); torch.save(artifact, destination)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
