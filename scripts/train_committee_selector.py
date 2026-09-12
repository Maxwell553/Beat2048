"""Train a conservative neural selector between two current-board committees."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bot2048.deep_rl import CommitteeSelector, FusionPolicy
from scripts.train_fusion_policy import features


def load_head(path, device):
    artifact = torch.load(path, map_location=device, weights_only=True)
    model = FusionPolicy(
        artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def committee_logits(models, x):
    return torch.stack([model(x[:, : model.inputs]) for model in models]).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", nargs="+", required=True)
    p.add_argument("--candidate", nargs="+", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=99960000)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.train) as archive:
        train = {name: archive[name] for name in archive.files}
    with np.load(args.validation) as archive:
        validation = {name: archive[name] for name in archive.files}
    base = [load_head(path, args.device) for path in args.base]
    candidate = [load_head(path, args.device) for path in args.candidate]
    inputs = features(train, np.arange(1), args.device).shape[1]
    model = CommitteeSelector(inputs, args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.001)
    order = np.arange(len(train["boards"]))
    best_gain = -1e30
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        model.train()
        total_loss = disagreement_count = 0
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset:offset + args.batch_size]
            x = features(train, ids, args.device)
            q = torch.as_tensor(train["targets"][ids], device=args.device)
            legal = q > -1e20
            with torch.no_grad():
                base_action = committee_logits(base, x).masked_fill(~legal, -1e9).argmax(1)
                candidate_action = committee_logits(candidate, x).masked_fill(~legal, -1e9).argmax(1)
                disagreement = base_action != candidate_action
                delta = q.gather(1, candidate_action[:, None]).squeeze(1) - q.gather(1, base_action[:, None]).squeeze(1)
            if not disagreement.any():
                continue
            logits = model(x[disagreement])
            target = (delta[disagreement] > 0).float()
            weight = 1 + (delta[disagreement].abs() / 1000).clamp(max=20)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none") * weight).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * int(disagreement.sum())
            disagreement_count += int(disagreement.sum())

        model.eval(); confidences = []; deltas = []
        with torch.no_grad():
            for offset in range(0, len(validation["boards"]), args.batch_size):
                ids = np.arange(offset, min(offset + args.batch_size, len(validation["boards"])))
                x = features(validation, ids, args.device)
                q = torch.as_tensor(validation["targets"][ids], device=args.device)
                legal = q > -1e20
                ba = committee_logits(base, x).masked_fill(~legal, -1e9).argmax(1)
                ca = committee_logits(candidate, x).masked_fill(~legal, -1e9).argmax(1)
                disagreement = ba != ca
                delta = q.gather(1, ca[:, None]).squeeze(1) - q.gather(1, ba[:, None]).squeeze(1)
                confidences.append(torch.sigmoid(model(x))[disagreement].cpu().numpy())
                deltas.append(delta[disagreement].cpu().numpy())
        confidence = np.concatenate(confidences)
        delta = np.concatenate(deltas)
        choices = []
        for threshold in np.arange(0.5, 0.991, 0.01):
            selected = confidence >= threshold
            gain = float(delta[selected].sum() / len(validation["boards"]))
            precision = float((delta[selected] > 0).mean()) if selected.any() else 0.0
            choices.append((gain, float(threshold), int(selected.sum()), precision))
        gain, threshold, selected, precision = max(choices)
        row = {
            "epoch": epoch, "loss": total_loss / max(1, disagreement_count),
            "train_disagreements": disagreement_count,
            "validation_disagreements": int(len(delta)), "threshold": threshold,
            "selected": selected, "positive_delta_precision": precision,
            "estimated_teacher_gain_per_state": gain,
        }
        if gain > best_gain:
            best_gain = gain
            artifact = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "metadata": {"inputs": inputs, "hidden": args.hidden,
                             "threshold": threshold, "config": vars(args), **row,
                             "search_at_inference": False},
            }
            torch.save(artifact, str(args.output) + ".tmp")
            Path(str(args.output) + ".tmp").replace(args.output)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
