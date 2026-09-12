"""Learn rare RL-teacher corrections on top of an unchanged neural vote policy."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvPolicy, ResidualSymmetryHead, encode
from scripts.train_conv_rl import augment, read
from scripts.train_symmetry_head import aligned_views


def loss_and_metrics(base, head, boards, q, device, correction_weight, l2_weight):
    target = torch.as_tensor(q, device=device)
    legal = target > -1e20
    views = aligned_views(base, boards, device)
    encoded = encode(boards, device)
    base_scores = head.vote_scores(views, legal).masked_fill(~legal, -1e9)
    logits = head(views, encoded, legal).masked_fill(~legal, -1e9)
    labels = target.argmax(1)
    base_actions = base_scores.argmax(1)
    corrections = labels != base_actions
    weights = 1 + corrections.float() * (correction_weight - 1)
    per_state = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    residual = logits - base_scores
    loss = (per_state * weights).sum() / weights.sum()
    loss = loss + l2_weight * residual.square().mean()
    return loss, {
        "accuracy": (logits.argmax(1) == labels).float().mean().item(),
        "base_accuracy": (base_actions == labels).float().mean().item(),
        "changed": (logits.argmax(1) != base_actions).float().mean().item(),
        "corrections": corrections.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--correction-weight", type=float, default=32.0)
    parser.add_argument("--l2-weight", type=float, default=0.01)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=2048)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = ConvPolicy(**artifact["metadata"].get("network_config", {})).to(args.device)
    base.load_state_dict(artifact["state_dict"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    head = ResidualSymmetryHead(args.hidden, args.scale).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    boards, targets = read(args.train)
    validation_boards, validation_targets = read(args.validation)
    order = np.arange(len(boards))
    started = time.time()
    metadata = {
        **artifact["metadata"],
        "symmetry_head": {
            "kind": "residual_vote",
            "hidden": args.hidden,
            "scale": args.scale,
        },
        "training_method": "residual correction of neural voting from learned RL values",
        "search_at_inference": False,
        "base_checkpoint": args.checkpoint,
        "base_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "head_training": vars(args),
    }
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        head.train()
        keys = ("loss", "accuracy", "base_accuracy", "changed", "corrections")
        sums = {key: 0.0 for key in keys}
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset : offset + args.batch_size]
            b, q = augment(boards[ids], targets[ids], rng)
            loss, metrics = loss_and_metrics(
                base, head, b, q, args.device, args.correction_weight, args.l2_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
            sums["loss"] += loss.item() * len(ids)
            for key, value in metrics.items():
                sums[key] += value * len(ids)
        head.eval()
        validation = {key: 0.0 for key in keys}
        with torch.no_grad():
            for offset in range(0, len(validation_boards), args.batch_size):
                b = validation_boards[offset : offset + args.batch_size]
                q = validation_targets[offset : offset + args.batch_size]
                loss, metrics = loss_and_metrics(
                    base, head, b, q, args.device, args.correction_weight, args.l2_weight
                )
                validation["loss"] += loss.item() * len(b)
                for key, value in metrics.items():
                    validation[key] += value * len(b)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / len(boards) for key, value in sums.items()},
            **{f"validation_{key}": value / len(validation_boards) for key, value in validation.items()},
            "seconds": time.time() - started,
        }
        checkpoint = {
            "state_dict": artifact["state_dict"],
            "head_state_dict": {name: value.cpu() for name, value in head.state_dict().items()},
            "metadata": {**metadata, **row},
        }
        destination = Path(args.output).with_name(f"{Path(args.output).stem}.epoch{epoch}.pt")
        torch.save(checkpoint, str(destination) + ".tmp")
        Path(str(destination) + ".tmp").replace(destination)
        torch.save(checkpoint, str(args.output) + ".tmp")
        Path(str(args.output) + ".tmp").replace(args.output)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
