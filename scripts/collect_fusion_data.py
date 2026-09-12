"""Precompute neural current-board features for a learned fusion policy."""

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
    p.add_argument("--expert", nargs="*", default=[], help="Additional frozen neural policy checkpoints")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--states", type=int, default=400000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.set_num_threads(2)
    all_boards, all_targets = read(args.data)
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(len(all_boards), min(args.states, len(all_boards)), replace=False)
    boards, targets = all_boards[ids], all_targets[ids]
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    experts = [ConvAgent(path, args.device) for path in args.expert]
    aligned_rows, votes_rows, probability_rows, sparse_rows, expert_rows = [], [], [], [], []
    for offset in range(0, len(boards), args.batch_size):
        current = boards[offset : offset + args.batch_size]
        legal = targets[offset : offset + len(current)] > -1e20
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
            encoded = encode(np.concatenate(variants), args.device)
            raw = cnn.model(encoded).reshape(8, len(current), 4)
            aligned = torch.stack([raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
            mask = torch.as_tensor(legal, dtype=torch.bool, device=args.device)
            masked = aligned.masked_fill(~mask[:, None], -1e9)
            choices = masked.argmax(2)
            votes = torch.stack([(choices == action).sum(1) for action in range(4)], 1)
            probabilities = torch.softmax(masked, 2).mean(1)
            expert_aligned = []
            for expert in experts:
                expert_raw = expert.model(encoded).reshape(8, len(current), 4)
                expert_view = torch.stack([expert_raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
                expert_view -= expert_view.mean(2, keepdim=True)
                expert_view /= expert_view.std(2, keepdim=True) + 1e-6
                expert_aligned.append(expert_view.cpu().numpy().astype(np.float16))
        aligned_np = aligned.cpu().numpy()
        aligned_np -= aligned_np.mean(2, keepdims=True)
        packed = np.sum(
            current.reshape(-1, 16).astype(np.uint64) << (4 * np.arange(16, dtype=np.uint64)),
            axis=1, dtype=np.uint64,
        )
        sparse_values = np.stack([model.values_batch(packed) for model in sparse], 1)
        sparse_values -= sparse_values.mean(2, keepdims=True)
        sparse_values /= sparse_values.std(2, keepdims=True) + 1e-6
        aligned_rows.append(aligned_np.astype(np.float16))
        votes_rows.append(votes.cpu().numpy().astype(np.uint8))
        probability_rows.append(probabilities.cpu().numpy().astype(np.float16))
        sparse_rows.append(sparse_values.astype(np.float16))
        if expert_aligned:
            expert_rows.append(np.stack(expert_aligned, 1))
    for model in sparse:
        model.close()
    payload = dict(boards=boards, aligned=np.concatenate(aligned_rows), votes=np.concatenate(votes_rows), probabilities=np.concatenate(probability_rows), sparse=np.concatenate(sparse_rows), targets=targets)
    if expert_rows:
        payload["experts"] = np.concatenate(expert_rows)
    np.savez_compressed(args.output, **payload)
    metadata = {"config": vars(args), "states": len(boards), "search_at_inference": False}
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
