"""Evaluate a learned fusion of direct CNN and sparse neural policies."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from bot2048.deep_rl import CommitteeSelector, ConvAgent, FusionPolicy, GatedFusionCorrection, encode
from bot2048.rl import RLAgent
from scripts.evaluate import wilson
from scripts.ppo_rl import BatchEnv


def fusion_actions(cnn, sparse, fusion, boards, legal, device, correction=None, correction_scale=1.0, gate=None, gate_threshold=0.5, experts=None, selector=None, selector_threshold=0.5, candidate_fusion=None, fusion_weights=None):
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
        encoded = encode(np.concatenate(variants), device)
        raw = cnn.model(encoded).reshape(8, len(boards), 4)
        aligned = torch.stack([raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
        mask = torch.as_tensor(legal, dtype=torch.bool, device=device)
        masked = aligned.masked_fill(~mask[:, None], -1e9)
        choices = masked.argmax(2)
        votes = torch.stack([(choices == action).sum(1) for action in range(4)], 1)
        probabilities = torch.softmax(masked, 2).mean(1)
        expert_features = []
        for expert in experts or []:
            expert_raw = expert.model(encoded).reshape(8, len(boards), 4)
            expert_view = torch.stack([expert_raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
            expert_view -= expert_view.mean(2, keepdim=True)
            expert_view /= expert_view.std(2, keepdim=True) + 1e-6
            expert_features.append(expert_view.cpu().numpy())
    aligned_np = aligned.cpu().numpy()
    aligned_np -= aligned_np.mean(2, keepdims=True)
    packed = np.sum(
        boards.reshape(-1, 16).astype(np.uint64) << (4 * np.arange(16, dtype=np.uint64)),
        axis=1, dtype=np.uint64,
    )
    sparse_values = np.stack([model.values_batch(packed) for model in sparse], 1)
    sparse_values -= sparse_values.mean(2, keepdims=True)
    sparse_values /= sparse_values.std(2, keepdims=True) + 1e-6
    onehot = torch.nn.functional.one_hot(
        torch.as_tensor(boards, device=device, dtype=torch.long).clamp(0, 15), 16
    ).flatten(1).float()
    rest = np.concatenate(
        [
            aligned_np.reshape(len(boards), -1),
            votes.cpu().numpy() / 8.0,
            probabilities.cpu().numpy(),
            sparse_values.reshape(len(boards), -1),
            *( [np.stack(expert_features, 1).reshape(len(boards), -1)] if expert_features else [] ),
        ], axis=1,
    ).astype(np.float32)
    with torch.no_grad():
        features = torch.cat([onehot, torch.as_tensor(rest, device=device)], 1)
        models = fusion if isinstance(fusion, (list, tuple)) else [fusion]
        # Committee heads may have been trained before additional frozen experts
        # were appended.  Each head consumes the prefix recorded in its own
        # checkpoint, allowing old and new current-board heads to vote together.
        component_logits = torch.stack(
            [model(features[:, : model.inputs]) for model in models]
        )
        if fusion_weights is None:
            logits = component_logits.mean(0)
        else:
            weights = torch.as_tensor(
                fusion_weights, device=device, dtype=component_logits.dtype
            )
            logits = (component_logits * weights[:, None, None]).sum(0) / weights.sum()
        if selector is not None and candidate_fusion is not None:
            candidate_models = candidate_fusion if isinstance(candidate_fusion, (list, tuple)) else [candidate_fusion]
            candidate_logits = torch.stack(
                [model(features[:, : model.inputs]) for model in candidate_models]
            ).mean(0)
            choose_candidate = torch.sigmoid(selector(features[:, : selector.inputs])) >= selector_threshold
            logits = torch.where(choose_candidate[:, None], candidate_logits, logits)
        if correction is not None:
            logits = logits + correction_scale * torch.tanh(correction(features))
        if gate is not None:
            confidence, gate_logits = gate(features)
            logits = torch.where((torch.sigmoid(confidence) >= gate_threshold)[:, None], gate_logits, logits)
        logits = logits.masked_fill(~mask, -1e9)
    return logits.argmax(1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True)
    p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--expert", nargs="*", default=[])
    p.add_argument("--fusion", nargs="+", required=True)
    p.add_argument("--fusion-weight", nargs="*", type=float)
    p.add_argument("--correction", help="Optional PPO residual fusion-head checkpoint")
    p.add_argument("--correction-scale", type=float, help="Override the saved residual scale")
    p.add_argument("--gate", help="Optional outcome-trained gated correction checkpoint")
    p.add_argument("--selector", help="Optional neural committee-selector checkpoint")
    p.add_argument("--selector-threshold", type=float)
    p.add_argument("--candidate-fusion", nargs="*", default=[])
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=97000000)
    p.add_argument("--device", default="mps")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    experts = [ConvAgent(path, args.device) for path in args.expert]
    fusion = []
    for path in args.fusion:
        artifact = torch.load(path, map_location=args.device, weights_only=True)
        model = FusionPolicy(artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]).to(args.device)
        model.load_state_dict(artifact["state_dict"]); model.eval(); fusion.append(model)
    if args.fusion_weight is not None and len(args.fusion_weight) != len(fusion):
        p.error("--fusion-weight must provide one weight per --fusion checkpoint")
    correction = None; correction_scale = 1.0
    if args.correction:
        artifact = torch.load(args.correction, map_location=args.device, weights_only=True)
        metadata = artifact["metadata"]
        correction = nn.Sequential(nn.Linear(metadata["inputs"], metadata["hidden"]), nn.ReLU(), nn.Linear(metadata["hidden"], 4)).to(args.device)
        correction.load_state_dict(artifact["correction_state_dict"]); correction.eval()
        correction_scale = metadata["scale"]
        if args.correction_scale is not None:
            correction_scale = args.correction_scale
    gate = None; gate_threshold = 0.5
    if args.gate:
        artifact = torch.load(args.gate, map_location=args.device, weights_only=True)
        metadata = artifact["metadata"]; gate = GatedFusionCorrection(metadata["inputs"], metadata["hidden"]).to(args.device)
        gate.load_state_dict(artifact["state_dict"]); gate.eval(); gate_threshold = metadata["threshold"]
    selector = None; selector_threshold = 0.5; candidate_fusion = []
    if args.selector:
        artifact = torch.load(args.selector, map_location=args.device, weights_only=True)
        metadata = artifact["metadata"]
        selector = CommitteeSelector(metadata["inputs"], metadata["hidden"]).to(args.device)
        selector.load_state_dict(artifact["state_dict"]); selector.eval()
        selector_threshold = metadata["threshold"] if args.selector_threshold is None else args.selector_threshold
        for path in args.candidate_fusion:
            candidate_artifact = torch.load(path, map_location=args.device, weights_only=True)
            model = FusionPolicy(candidate_artifact["metadata"]["inputs"], candidate_artifact["metadata"]["hidden"]).to(args.device)
            model.load_state_dict(candidate_artifact["state_dict"]); model.eval(); candidate_fusion.append(model)
    env = BatchEnv(args.games, args.seed)
    finished = np.zeros(args.games, dtype=bool); wins = np.zeros(args.games, dtype=bool)
    moves = np.zeros(args.games, dtype=np.int32); started = time.time()
    while not finished.all():
        selected = fusion_actions(cnn, sparse, fusion, env.boards, env.legal, args.device, correction, correction_scale, gate, gate_threshold, experts, selector, selector_threshold, candidate_fusion, args.fusion_weight)
        env.step(selected); active = ~finished; moves[active] += 1
        ended = active & (env.outcomes != 0); wins[ended] = env.outcomes[ended] == 1; finished[ended] = True
        if moves.max() >= 20000: raise RuntimeError("game reached move cap")
    env.close()
    for model in sparse: model.close()
    count = int(wins.sum())
    result = {
        "config": vars(args),
        "hashes": {
            "cnn": hashlib.sha256(Path(args.cnn).read_bytes()).hexdigest(),
            "fusion": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.fusion},
            "correction": hashlib.sha256(Path(args.correction).read_bytes()).hexdigest() if args.correction else None,
            "gate": hashlib.sha256(Path(args.gate).read_bytes()).hexdigest() if args.gate else None,
            "selector": hashlib.sha256(Path(args.selector).read_bytes()).hexdigest() if args.selector else None,
            "candidate_fusion": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.candidate_fusion},
            "sparse": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.sparse},
            "experts": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.expert},
        },
        "environment": "native standard 2048; two initial tiles; 90% 2 / 10% 4",
        "search_at_inference": False,
        "summary": {"games": args.games, "wins": count, "wilson_95": wilson(count, args.games), "mean_moves": float(moves.mean()), "seconds": time.time() - started},
        "games": [{"seed": args.seed + i, "won": bool(wins[i]), "moves": int(moves[i])} for i in range(args.games)],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
