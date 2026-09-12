"""Distill an RL-trained value teacher into a direct, search-free neural policy."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from bot2048.deep_rl import (
    ConvPolicy,
    DensePolicy,
    DirectionalPolicy,
    EquivariantConvPolicy,
    LineEmbeddingPolicy,
    CanonicalLinePolicy,
    TransformerPolicy,
    encode,
    policy_from_config,
)

RECORD = np.dtype([("board", "<u8"), ("q", "<f4", (4,))])


def read(path):
    raw = np.fromfile(path, dtype=RECORD)
    boards = (
        ((raw["board"][:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 15)
        .astype(np.uint8)
        .reshape(-1, 4, 4)
    )
    return boards, raw["q"].copy()


def augment(boards, q, rng):
    boards = boards.copy()
    q = q.copy()
    choices = rng.integers(0, 8, len(boards))
    for transform in range(8):
        mask = choices == transform
        k = transform % 4
        boards[mask] = np.rot90(boards[mask], k, axes=(1, 2))
        mapping = (np.arange(4) - k) % 4
        if transform >= 4:
            boards[mask] = boards[mask, :, ::-1]
            mapping = (-mapping) % 4
        q[mask] = q[mask][:, np.argsort(mapping)]
    return boards, q


def ensemble_logits(model, boards, device, aggregation="mean", vote_temperature=0.1):
    """Differentiable D4 averaging in the original board's action coordinates."""
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
    raw = model(encode(np.concatenate(variants), device)).reshape(8, len(boards), 4)
    aligned = torch.stack([raw[i][:, mapping] for i, mapping in enumerate(mappings)])
    if aggregation == "mean":
        return aligned.mean(0)
    if aggregation == "probability":
        # A low temperature approaches one vote per view while retaining gradients.
        return torch.log_softmax(aligned / vote_temperature, dim=2).logsumexp(0)
    raise ValueError(f"unknown ensemble aggregation: {aggregation}")


def loss_and_accuracy(
    model,
    b,
    q,
    device,
    temperature,
    regret_weight=0,
    hard_targets=False,
    late_state_weight=0,
    ensemble_training=False,
    margin_weight=0,
    ensemble_aggregation="mean",
    vote_temperature=0.1,
):
    x = encode(b, device)
    target = torch.tensor(q, device=device)
    legal = target > -1e20
    regret = (target.max(1, keepdim=True).values - target).clamp(0, 50000) / 1000
    target = (target - target.max(1, keepdim=True).values) / temperature
    logits = (
        ensemble_logits(model, b, device, ensemble_aggregation, vote_temperature)
        if ensemble_training
        else model(x)
    ).masked_fill(~legal, -1e9)
    if hard_targets:
        per_state = torch.nn.functional.cross_entropy(
            logits, target.argmax(1), reduction="none"
        )
    else:
        probabilities = torch.softmax(target, dim=1)
        per_state = -(probabilities * torch.log_softmax(logits, dim=1)).sum(1)
    if regret_weight:
        per_state = per_state + regret_weight * (
            torch.softmax(logits, dim=1) * regret
        ).sum(1)
    if margin_weight:
        best = target.argmax(1)
        best_logit = logits.gather(1, best[:, None])
        violations = torch.relu(1 + logits - best_logit).masked_fill(~legal, 0)
        violations.scatter_(1, best[:, None], 0)
        per_state = per_state + margin_weight * (violations * regret).sum(1)
    if late_state_weight:
        largest = torch.as_tensor(
            np.asarray(b).max(axis=(1, 2)), device=device, dtype=torch.float32
        )
        # Rank 9 is 512. Weight rises smoothly through the decisive 512/1024 phase.
        weights = 1 + late_state_weight * ((largest - 8).clamp(0, 2) / 2)
        loss = (per_state * weights).sum() / weights.sum()
    else:
        loss = per_state.mean()
    agreement = (logits.argmax(1) == target.argmax(1)).float().mean()
    return loss, agreement


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/rl_teacher_train.bin")
    p.add_argument("--validation", default="data/rl_teacher_validation.bin")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="mps")
    p.add_argument("--temperature", type=float, default=200)
    p.add_argument("--regret-weight", type=float, default=0)
    p.add_argument(
        "--margin-weight",
        type=float,
        default=0,
        help="Cost-sensitive teacher-ranking hinge weight; regret is scaled by 1/1000",
    )
    p.add_argument("--hard-targets", action="store_true")
    p.add_argument(
        "--ensemble-training",
        action="store_true",
        help="Optimize the same eight-orientation averaged logits used in deployment",
    )
    p.add_argument(
        "--ensemble-aggregation",
        choices=["mean", "probability"],
        default="mean",
        help="Differentiable aggregation used with --ensemble-training",
    )
    p.add_argument(
        "--vote-temperature",
        type=float,
        default=0.1,
        help="Sharpness of probability aggregation; lower values approach voting",
    )
    p.add_argument(
        "--late-state-weight",
        type=float,
        default=0,
        help="Additional loss weight, reaching this value at boards with a 1024 tile",
    )
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument(
        "--adapter-only",
        action="store_true",
        help="Freeze the base policy and optimize only a configured residual adapter",
    )
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--output", default="models/rl/conv_policy.pt")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--architecture",
        choices=["conv", "dense", "directional", "equivariant_conv", "transformer", "line_embedding", "canonical_line"],
        default="conv",
    )
    p.add_argument("--initial-output")
    p.add_argument(
        "--resume", help="Initialize from an existing policy; optimizer restarts"
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    boards, targets = read(a.train)
    vb, vq = read(a.validation)
    if a.architecture == "dense":
        model = DensePolicy(width=a.width, blocks=a.blocks)
    elif a.architecture == "directional":
        model = DirectionalPolicy(width=a.width, hidden=a.hidden, blocks=a.blocks)
    elif a.architecture == "equivariant_conv":
        model = EquivariantConvPolicy(
            width=a.width, blocks=a.blocks, hidden=a.hidden,
            vote_temperature=a.vote_temperature
        )
    elif a.architecture == "transformer":
        model = TransformerPolicy(
            width=a.width, hidden=a.hidden, blocks=a.blocks, heads=a.heads
        )
    elif a.architecture == "line_embedding":
        model = LineEmbeddingPolicy(
            embedding=a.width, hidden=a.hidden, blocks=a.blocks
        )
    elif a.architecture == "canonical_line":
        model = CanonicalLinePolicy(
            embedding=a.width, hidden=a.hidden, blocks=a.blocks
        )
    else:
        model = ConvPolicy(width=a.width, blocks=a.blocks, hidden=a.hidden)
    model = model.to(a.device)
    if a.resume:
        previous = torch.load(a.resume, map_location=a.device, weights_only=True)
        model = policy_from_config(
            previous["metadata"].get("network_config", {})
        ).to(a.device)
        model.load_state_dict(previous["state_dict"])
    if a.adapter_only:
        if model.adapter is None:
            p.error("--adapter-only requires a checkpoint with an adapter")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("adapter.")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=a.learning_rate,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, a.epochs, eta_min=a.learning_rate * 0.1
    )
    metadata = {
        "architecture": (
            "dense-residual-policy4"
            if isinstance(model, DensePolicy)
            else "directional-convolutional-policy4"
            if isinstance(model, DirectionalPolicy)
            else "board-token-transformer-policy4"
            if isinstance(model, TransformerPolicy)
            else "configurable-residual-policy4"
        ),
        "network_config": model.config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_method": "distillation of temporal-difference RL teacher; no expectimax",
        "search_at_inference": False,
        "config": vars(a),
        "train_states": len(boards),
        "validation_states": len(vb),
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        initial_output = a.initial_output or str(
            Path(a.output).with_name(Path(a.output).stem + "_initial.pt")
        )
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "metadata": {**metadata, "trained": False},
            },
            initial_output,
        )
    order = np.arange(len(boards))
    best = float("inf")
    history = []
    started = time.time()
    for epoch in range(a.epochs):
        model.train()
        rng.shuffle(order)
        s_loss = s_accuracy = count = 0
        for offset in range(0, len(order), a.batch_size):
            ids = order[offset : offset + a.batch_size]
            b, q = boards[ids], targets[ids]
            if not a.ensemble_training:
                b, q = augment(b, q, rng)
            loss, accuracy = loss_and_accuracy(
                model,
                b,
                q,
                a.device,
                a.temperature,
                a.regret_weight,
                a.hard_targets,
                a.late_state_weight,
                a.ensemble_training,
                a.margin_weight,
                a.ensemble_aggregation,
                a.vote_temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            s_loss += loss.item() * len(ids)
            s_accuracy += accuracy.item() * len(ids)
            count += len(ids)
        model.eval()
        v_loss = v_accuracy = v_count = 0
        with torch.no_grad():
            for offset in range(0, len(vb), a.batch_size):
                b, q = (
                    vb[offset : offset + a.batch_size],
                    vq[offset : offset + a.batch_size],
                )
                loss, accuracy = loss_and_accuracy(
                    model,
                    b,
                    q,
                    a.device,
                    a.temperature,
                    a.regret_weight,
                    a.hard_targets,
                    a.late_state_weight,
                    a.ensemble_training,
                    a.margin_weight,
                    a.ensemble_aggregation,
                    a.vote_temperature,
                )
                v_loss += loss.item() * len(b)
                v_accuracy += accuracy.item() * len(b)
                v_count += len(b)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": s_loss / count,
            "train_accuracy": s_accuracy / count,
            "validation_loss": v_loss / v_count,
            "validation_accuracy": v_accuracy / v_count,
            "seconds": time.time() - started,
        }
        history.append(row)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "metadata": {**metadata, "trained": True, **row},
                },
                str(a.output) + ".tmp",
            )
            Path(str(a.output) + ".tmp").replace(a.output)
        if a.save_every_epoch:
            snapshot = Path(a.output).with_name(
                f"{Path(a.output).stem}.epoch{epoch + 1}.pt"
            )
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "metadata": {**metadata, "trained": True, **row},
                },
                str(snapshot) + ".tmp",
            )
            Path(str(snapshot) + ".tmp").replace(snapshot)
        print(json.dumps(row), flush=True)
        (Path("results/rl") / (Path(a.output).stem + "_training.json")).write_text(
            json.dumps({"metadata": metadata, "history": history}, indent=2)
        )


if __name__ == "__main__":
    main()
