"""Record a saved direct neural policy, including a complete replayable trace."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent
from bot2048.env import Game2048
from scripts.animate import frame


def record(checkpoint, seed, ensemble, voting):
    agent = ConvAgent(checkpoint, ensemble=ensemble, voting=voting)
    game = Game2048(seed)
    trajectory = [{"board": game.board.tolist(), "score": 0, "move": 0}]
    while not game.done:
        if game.moves >= 20000:
            raise RuntimeError(
                "Recording truncated; refusing to label it a finished game"
            )
        action = agent.act(game.board)
        transition = game.step(action)
        if not transition.changed:
            raise RuntimeError("Policy selected an invalid action")
        trajectory.append(
            {
                "board": game.board.tolist(),
                "score": game.score,
                "move": game.moves,
                "action": action,
            }
        )
    replay = Game2048(seed)
    for state in trajectory[1:]:
        replay.step(state["action"])
        np.testing.assert_array_equal(replay.board, state["board"])
        assert replay.score == state["score"]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_metadata": agent.metadata,
        "seed": seed,
        "symmetry_ensemble": ensemble,
        "symmetry_voting": voting,
        "search_at_inference": False,
        "replay_verified": True,
        "won": game.won,
        "score": game.score,
        "moves": game.moves,
        "max_tile": 2 ** int(game.board.max()),
        "trajectory": trajectory,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Output prefix without extension")
    p.add_argument("--title", default="Neural policy candidate")
    p.add_argument("--single-orientation", action="store_true")
    p.add_argument(
        "--voting",
        action="store_true",
        help="Use majority voting across the eight neural symmetry predictions",
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    ensemble = not a.single_orientation
    if a.voting and not ensemble:
        p.error("--voting cannot be combined with --single-orientation")
    row = record(a.checkpoint, a.seed, ensemble, a.voting)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    states = row["trajectory"]
    indices = np.unique(
        np.linspace(0, len(states) - 1, min(150, len(states)), dtype=int)
    )
    subtitle = "Neural decisions only • accelerated replay"
    frames = [frame(states[i], a.title, subtitle, seed=a.seed) for i in indices]
    final = frame(states[-1], a.title, subtitle, True, row["won"], a.seed)
    frames.append(final)
    frames[0].save(
        out.with_suffix(".gif"),
        save_all=True,
        append_images=frames[1:],
        duration=[90] * (len(frames) - 1) + [2200],
        loop=0,
        optimize=True,
    )
    final.save(out.with_name(out.name + "_final.png"))
    print(
        json.dumps(
            {
                k: v
                for k, v in row.items()
                if k not in ("trajectory", "checkpoint_metadata")
            }
        )
    )


if __name__ == "__main__":
    main()
