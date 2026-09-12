"""Screen CNN symmetry voting with a direct sparse neural tie-breaker."""

import argparse
import json

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, encode
from bot2048.rl import RLAgent
from scripts.evaluate import wilson
from scripts.ppo_rl import BatchEnv
from scripts.train_tie_gate import TieGate


def actions(
    cnn,
    sparse,
    boards,
    legal,
    minimum_sparse_margin=-1e30,
    maximum_empties=16,
    minimum_rank=0,
    minimum_top_votes=0,
    maximum_tied_actions=4,
    sparse_aggregation="value",
    gate=None,
    gate_threshold=0.5,
    maximum_vote_deficit=0,
):
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
        raw = cnn.model(encode(np.concatenate(variants), cnn.device)).reshape(
            8, len(boards), 4
        )
        aligned = torch.stack(
            [raw[index][:, mapping] for index, mapping in enumerate(mappings)]
        )
        mask = torch.as_tensor(legal, dtype=torch.bool, device=cnn.device)
        choices = aligned.masked_fill(~mask[None], -1e9).argmax(2)
        votes = torch.stack(
            [(choices == action).sum(0) for action in range(4)], dim=1
        ).cpu().numpy()
        probabilities = torch.softmax(
            aligned.masked_fill(~mask[None], -1e9), dim=2
        ).mean(0).cpu().numpy()
        aligned_numpy = aligned.cpu().numpy()
    packed = np.sum(
        boards.reshape(-1, 16).astype(np.uint64)
        << (4 * np.arange(16, dtype=np.uint64)),
        axis=1,
        dtype=np.uint64,
    )
    sparse_models = sparse if isinstance(sparse, (list, tuple)) else [sparse]
    sparse_predictions = np.stack(
        [model.values_batch(packed) for model in sparse_models]
    )
    sparse_values = sparse_predictions.mean(0)
    result = np.empty(len(boards), dtype=np.int64)
    gate_features, gate_indices, gate_sparse_actions = [], [], []
    for index in range(len(boards)):
        legal_actions = np.flatnonzero(legal[index])
        counts = votes[index, legal_actions]
        tied = legal_actions[counts >= counts.max() - maximum_vote_deficit]
        cnn_action = tied[np.argmax(probabilities[index, tied])]
        sparse_order = tied[np.argsort(sparse_values[index, tied])]
        if sparse_aggregation == "vote" and len(sparse_models) > 1:
            member_actions = np.asarray(
                [tied[np.argmax(values[index, tied])] for values in sparse_predictions]
            )
            member_votes = np.asarray([(member_actions == a).sum() for a in tied])
            finalists = tied[member_votes == member_votes.max()]
            sparse_action = finalists[np.argmax(sparse_values[index, finalists])]
        else:
            sparse_action = sparse_order[-1]
        sparse_margin = (
            sparse_values[index, sparse_order[-1]]
            - sparse_values[index, sparse_order[-2]]
            if len(sparse_order) > 1
            else 1e30
        )
        use_sparse = (
            sparse_margin >= minimum_sparse_margin
            and int((boards[index] == 0).sum()) <= maximum_empties
            and int(boards[index].max()) >= minimum_rank
            and int(counts.max()) >= minimum_top_votes
            and len(tied) <= maximum_tied_actions
        )
        result[index] = sparse_action if use_sparse else cnn_action
        if gate is not None and cnn_action != sparse_action:
            view_logits = aligned_numpy[:, index].copy()
            view_logits -= view_logits.mean(1, keepdims=True)
            sparse_logits = sparse_predictions[:, index].copy()
            sparse_logits -= sparse_logits.mean(1, keepdims=True)
            sparse_logits /= sparse_logits.std(1, keepdims=True) + 1e-6
            action_features = np.zeros(8, dtype=np.float32)
            action_features[cnn_action] = 1
            action_features[4 + sparse_action] = 1
            gate_features.append(
                np.concatenate(
                    [
                        boards[index].reshape(-1) / 15.0,
                        view_logits.reshape(-1),
                        votes[index] / 8.0,
                        probabilities[index],
                        sparse_logits.reshape(-1),
                        action_features,
                        [boards[index].max() / 15.0, (boards[index] == 0).sum() / 16.0],
                    ]
                ).astype(np.float32)
            )
            gate_indices.append(index)
            gate_sparse_actions.append(sparse_action)
    if gate_features:
        model, gate_device = gate
        with torch.no_grad():
            probabilities_gate = torch.sigmoid(
                model(torch.as_tensor(np.asarray(gate_features), device=gate_device))
            ).cpu().numpy()
        for index, sparse_action, probability in zip(
            gate_indices, gate_sparse_actions, probabilities_gate
        ):
            if probability >= gate_threshold:
                result[index] = sparse_action
            else:
                # Gate arbitration supersedes the static sparse rule.
                legal_actions = np.flatnonzero(legal[index])
                counts = votes[index, legal_actions]
                tied = legal_actions[counts == counts.max()]
                result[index] = tied[np.argmax(probabilities[index, tied])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", required=True)
    parser.add_argument("--sparse", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90000000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--minimum-sparse-margin", type=float, default=-1e30)
    parser.add_argument("--maximum-empties", type=int, default=16)
    parser.add_argument("--minimum-rank", type=int, default=0)
    parser.add_argument("--minimum-top-votes", type=int, default=0)
    parser.add_argument("--maximum-tied-actions", type=int, default=4)
    parser.add_argument(
        "--sparse-aggregation", choices=["value", "vote"], default="value"
    )
    parser.add_argument("--gate")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--maximum-vote-deficit", type=int, default=0)
    args = parser.parse_args()
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    gate = None
    if args.gate:
        artifact = torch.load(args.gate, map_location=args.device, weights_only=True)
        gate_model = TieGate(
            artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]
        ).to(args.device)
        gate_model.load_state_dict(artifact["state_dict"])
        gate_model.eval()
        gate = (gate_model, args.device)
    env = BatchEnv(args.games, args.seed)
    finished = np.zeros(args.games, dtype=bool)
    wins = np.zeros(args.games, dtype=bool)
    while not finished.all():
        env.step(
            actions(
                cnn,
                sparse,
                env.boards,
                env.legal,
                args.minimum_sparse_margin,
                args.maximum_empties,
                args.minimum_rank,
                args.minimum_top_votes,
                args.maximum_tied_actions,
                args.sparse_aggregation,
                gate,
                args.gate_threshold,
                args.maximum_vote_deficit,
            )
        )
        ended = (~finished) & (env.outcomes != 0)
        wins[ended] = env.outcomes[ended] == 1
        finished[ended] = True
    count = int(wins.sum())
    print(json.dumps({"games": args.games, "wins": count, "wilson_95": wilson(count, args.games)}))
    for model in sparse:
        model.close()
    env.close()


if __name__ == "__main__":
    main()
