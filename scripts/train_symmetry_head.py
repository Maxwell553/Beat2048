"""Train a neural combiner over eight frozen current-board policy views."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvPolicy, SymmetryHead, encode
from scripts.train_conv_rl import augment, read


def aligned_views(model, boards, device):
    variants, mappings = [], []
    for reflection in (False, True):
        for k in range(4):
            variant = np.rot90(boards, k, axes=(1, 2))
            mapping = (np.arange(4) - k) % 4
            if reflection:
                variant = variant[:, :, ::-1]
                mapping = (-mapping) % 4
            variants.append(variant.copy())
            mappings.append(mapping.tolist())
    with torch.no_grad():
        raw = model(encode(np.concatenate(variants), device)).reshape(
            8, len(boards), 4
        )
    return torch.stack(
        [raw[index][:, mapping] for index, mapping in enumerate(mappings)], dim=1
    )


def batch_loss(base, head, boards, q, device, late_weight):
    target = torch.as_tensor(q, device=device)
    legal = target > -1e20
    views = aligned_views(base, boards, device)
    logits = head(views, encode(boards, device)).masked_fill(~legal, -1e9)
    per_state = torch.nn.functional.cross_entropy(
        logits, target.argmax(1), reduction="none"
    )
    if late_weight:
        largest = torch.as_tensor(
            boards.max(axis=(1, 2)), device=device, dtype=torch.float32
        )
        weights = 1 + late_weight * ((largest - 8).clamp(0, 2) / 2)
        loss = (per_state * weights).sum() / weights.sum()
    else:
        loss = per_state.mean()
    accuracy = (logits.argmax(1) == target.argmax(1)).float().mean()
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--late-state-weight", type=float, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=2026)
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
    head = SymmetryHead(args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    boards, targets = read(args.train)
    validation_boards, validation_targets = read(args.validation)
    order = np.arange(len(boards))
    started = time.time()
    metadata = {
        **artifact["metadata"],
        "symmetry_head": {"hidden": args.hidden},
        "training_method": "RL-teacher distillation into a learned neural symmetry combiner",
        "search_at_inference": False,
        "base_checkpoint": args.checkpoint,
        "base_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "head_training": vars(args),
    }
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        head.train()
        train_loss = train_accuracy = 0.0
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset : offset + args.batch_size]
            batch_boards, batch_targets = augment(
                boards[ids], targets[ids], rng
            )
            loss, accuracy = batch_loss(
                base,
                head,
                batch_boards,
                batch_targets,
                args.device,
                args.late_state_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ids)
            train_accuracy += accuracy.item() * len(ids)
        head.eval()
        validation_loss = validation_accuracy = 0.0
        with torch.no_grad():
            for offset in range(0, len(validation_boards), args.batch_size):
                batch_boards = validation_boards[offset : offset + args.batch_size]
                batch_targets = validation_targets[offset : offset + args.batch_size]
                loss, accuracy = batch_loss(
                    base,
                    head,
                    batch_boards,
                    batch_targets,
                    args.device,
                    args.late_state_weight,
                )
                validation_loss += loss.item() * len(batch_boards)
                validation_accuracy += accuracy.item() * len(batch_boards)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(boards),
            "train_accuracy": train_accuracy / len(boards),
            "validation_loss": validation_loss / len(validation_boards),
            "validation_accuracy": validation_accuracy / len(validation_boards),
            "seconds": time.time() - started,
        }
        checkpoint = {
            "state_dict": artifact["state_dict"],
            "head_state_dict": {name: value.cpu() for name, value in head.state_dict().items()},
            "metadata": {**metadata, **row},
        }
        destination = Path(args.output).with_name(
            f"{Path(args.output).stem}.epoch{epoch}.pt"
        )
        torch.save(checkpoint, str(destination) + ".tmp")
        Path(str(destination) + ".tmp").replace(destination)
        torch.save(checkpoint, str(args.output) + ".tmp")
        Path(str(args.output) + ".tmp").replace(args.output)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
