"""Train a high-precision neural gate for rare RL policy corrections."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvPolicy, GatedSymmetryHead, ResidualSymmetryHead, encode
from scripts.train_conv_rl import augment, read
from scripts.train_symmetry_head import aligned_views


def batch(base, head, boards, q, device, positive_weight):
    target = torch.as_tensor(q, device=device)
    legal = target > -1e20
    views = aligned_views(base, boards, device)
    encoded = encode(boards, device)
    base_actions = ResidualSymmetryHead.vote_scores(views, legal).argmax(1)
    labels = target.argmax(1)
    positive = labels != base_actions
    gate, actions = head.raw(views, encoded)
    gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        gate,
        positive.float(),
        pos_weight=torch.tensor(float(positive_weight), device=device),
    )
    if positive.any():
        action_loss = torch.nn.functional.cross_entropy(
            actions[positive].masked_fill(~legal[positive], -1e9), labels[positive]
        )
        action_accuracy = (actions[positive].argmax(1) == labels[positive]).float().mean()
    else:
        action_loss = actions.sum() * 0
        action_accuracy = torch.tensor(0.0, device=device)
    loss = gate_loss + action_loss
    probabilities = torch.sigmoid(gate)
    return loss, {
        "gate_loss": gate_loss.item(),
        "action_loss": action_loss.item(),
        "action_accuracy": action_accuracy.item(),
        "positive_rate": positive.float().mean().item(),
        "positive_probability": probabilities[positive].mean().item() if positive.any() else 0,
        "negative_probability": probabilities[~positive].mean().item() if (~positive).any() else 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--positive-weight", type=float, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = ConvPolicy(**artifact["metadata"].get("network_config", {})).to(args.device)
    base.load_state_dict(artifact["state_dict"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    head = GatedSymmetryHead(args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    boards, targets = read(args.train)
    vb, vq = read(args.validation)
    order = np.arange(len(boards))
    started = time.time()
    metadata = {
        **artifact["metadata"],
        "symmetry_head": {"kind": "gated_vote", "hidden": args.hidden, "threshold": 0.5},
        "training_method": "neural risk gate and correction action from learned RL values",
        "search_at_inference": False,
        "base_checkpoint": args.checkpoint,
        "base_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "head_training": vars(args),
    }
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        totals = {}
        head.train()
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset : offset + args.batch_size]
            b, q = augment(boards[ids], targets[ids], rng)
            loss, metrics = batch(base, head, b, q, args.device, args.positive_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
            for key, value in {"loss": loss.item(), **metrics}.items():
                totals[key] = totals.get(key, 0) + value * len(ids)
        validation = {}
        head.eval()
        with torch.no_grad():
            for offset in range(0, len(vb), args.batch_size):
                b, q = vb[offset : offset + args.batch_size], vq[offset : offset + args.batch_size]
                loss, metrics = batch(base, head, b, q, args.device, args.positive_weight)
                for key, value in {"loss": loss.item(), **metrics}.items():
                    validation[key] = validation.get(key, 0) + value * len(b)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / len(boards) for k, v in totals.items()},
            **{f"validation_{k}": v / len(vb) for k, v in validation.items()},
            "seconds": time.time() - started,
        }
        checkpoint = {
            "state_dict": artifact["state_dict"],
            "head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
            "metadata": {**metadata, **row},
        }
        if epoch % args.save_every == 0 or epoch == args.epochs:
            destination = Path(args.output).with_name(f"{Path(args.output).stem}.epoch{epoch}.pt")
            torch.save(checkpoint, str(destination) + ".tmp")
            Path(str(destination) + ".tmp").replace(destination)
        torch.save(checkpoint, str(args.output) + ".tmp")
        Path(str(args.output) + ".tmp").replace(args.output)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
