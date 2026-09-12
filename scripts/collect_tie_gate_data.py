"""Build current-board features for learning CNN-versus-sparse tie arbitration."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, encode
from bot2048.rl import RLAgent
from scripts.train_conv_rl import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True)
    p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.set_num_threads(2)
    boards, teacher = read(args.data)
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    feature_rows, labels, regrets = [], [], []
    for offset in range(0, len(boards), args.batch_size):
        current = boards[offset : offset + args.batch_size]
        targets = teacher[offset : offset + args.batch_size]
        legal = targets > -1e20
        variants, mappings = [], []
        for reflection in (False, True):
            for k in range(4):
                variant = np.rot90(current, k, axes=(1, 2))
                mapping = (np.arange(4) - k) % 4
                if reflection:
                    variant = variant[:, :, ::-1]
                    mapping = (-mapping) % 4
                variants.append(variant.copy())
                mappings.append(mapping.tolist())
        with torch.no_grad():
            raw = cnn.model(encode(np.concatenate(variants), args.device)).reshape(
                8, len(current), 4
            )
            aligned = torch.stack(
                [raw[i][:, mapping] for i, mapping in enumerate(mappings)]
            )
            mask = torch.as_tensor(legal, dtype=torch.bool, device=args.device)
            masked = aligned.masked_fill(~mask[None], -1e9)
            choices = masked.argmax(2)
            votes = torch.stack(
                [(choices == action).sum(0) for action in range(4)], dim=1
            ).cpu().numpy()
            probabilities = torch.softmax(masked, 2).mean(0).cpu().numpy()
            aligned_np = aligned.cpu().numpy()
        packed = np.sum(
            current.reshape(-1, 16).astype(np.uint64)
            << (4 * np.arange(16, dtype=np.uint64)),
            axis=1,
            dtype=np.uint64,
        )
        sparse_values = np.stack([model.values_batch(packed) for model in sparse])
        mean_sparse = sparse_values.mean(0)
        for index in range(len(current)):
            available = np.flatnonzero(legal[index])
            tied = available[votes[index, available] == votes[index, available].max()]
            if len(tied) < 2:
                continue
            cnn_action = tied[np.argmax(probabilities[index, tied])]
            sparse_action = tied[np.argmax(mean_sparse[index, tied])]
            if cnn_action == sparse_action:
                continue
            view_logits = aligned_np[:, index]
            view_logits = view_logits - view_logits.mean(1, keepdims=True)
            sparse_logits = sparse_values[:, index]
            sparse_logits = sparse_logits - sparse_logits.mean(1, keepdims=True)
            sparse_logits /= sparse_logits.std(1, keepdims=True) + 1e-6
            action_features = np.zeros(8, dtype=np.float32)
            action_features[cnn_action] = 1
            action_features[4 + sparse_action] = 1
            feature_rows.append(
                np.concatenate(
                    [
                        current[index].reshape(-1) / 15.0,
                        view_logits.reshape(-1),
                        votes[index] / 8.0,
                        probabilities[index],
                        sparse_logits.reshape(-1),
                        action_features,
                        [current[index].max() / 15.0, (current[index] == 0).sum() / 16.0],
                    ]
                ).astype(np.float32)
            )
            difference = float(targets[index, sparse_action] - targets[index, cnn_action])
            labels.append(difference > 0)
            regrets.append(abs(difference))
    for model in sparse:
        model.close()
    features = np.asarray(feature_rows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    regrets = np.asarray(regrets, dtype=np.float32)
    np.savez_compressed(args.output, features=features, labels=labels, regrets=regrets)
    metadata = {
        "config": vars(args),
        "states": len(features),
        "features": features.shape[1],
        "prefer_sparse_fraction": float(labels.mean()),
        "mean_absolute_teacher_difference": float(regrets.mean()),
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
