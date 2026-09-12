"""Play a reproducible game: python3 -m bot2048 --seed 42."""

import argparse
import json
from .env import Game2048, ACTIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["value", "fusion", "conv", "rl", "random", "neural", "search", "hybrid"],
        default="value",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint", help="Optional checkpoint override for the selected mode"
    )
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=0.0001)
    p.add_argument("--neural-weight", type=float, default=0.02)
    p.add_argument("--board", help="JSON 4x4 array of tile values, 0 for empty")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--single-orientation",
        action="store_true",
        help="Disable neural symmetry averaging",
    )
    a = p.parse_args()
    import numpy as np
    import torch

    torch.set_num_threads(1)
    board = None
    if a.board:
        values = np.asarray(json.loads(a.board))
        if (
            values.shape != (4, 4)
            or np.any(values < 0)
            or np.any(values != values.astype(np.int64))
        ):
            p.error("board must contain 4x4 nonnegative integer tile values")
        board = np.zeros((4, 4), dtype=int)
        for index, v in np.ndenumerate(values):
            v = int(v)
            if v and (v < 2 or v & (v - 1)):
                p.error("nonempty tiles must be powers of two >= 2")
            board[index] = v.bit_length() - 1 if v else 0
    if a.mode == "value":
        from .value_agent import ValueEnsembleAgent

        agent = ValueEnsembleAgent(
            a.checkpoint or "models/rl/final_value_policy.json"
        )
    elif a.mode == "fusion":
        from .fusion import FusionAgent

        agent = FusionAgent(a.checkpoint or "models/rl/best_policy.json")
    elif a.mode == "conv":
        from .deep_rl import ConvAgent

        agent = ConvAgent(
            a.checkpoint or "models/rl/best_policy.pt",
            ensemble=not a.single_orientation,
            voting=not a.single_orientation,
        )
    elif a.mode == "rl":
        from .rl import RLAgent

        agent = RLAgent(a.checkpoint or "models/rl/direct_q.bin")
    else:
        # Legacy benchmark modes are isolated from the default direct-policy path.
        from .model import Agent

        agent = Agent(
            (
                (a.checkpoint or "models/final_policy.pt")
                if a.mode in ("neural", "hybrid")
                else None
            ),
            a.mode,
            a.seed + 314159,
            a.depth,
            a.cutoff,
            a.neural_weight,
        )
    g = Game2048(a.seed, board)
    while not g.done:
        move = agent.act(g.board)
        g.step(move)
        if not a.quiet:
            print(f"\nMove {g.moves}: {ACTIONS[move]} | Score {g.score}")
            for row in g.board:
                print(" ".join(f"{2**int(v) if v else 0:5}" for v in row))
    if hasattr(agent, "close"):
        agent.close()
    print(
        json.dumps(
            {
                "seed": a.seed,
                "mode": a.mode,
                "won": g.won,
                "score": g.score,
                "moves": g.moves,
                "max_tile": 2 ** int(g.board.max()),
            }
        )
    )


if __name__ == "__main__":
    main()
