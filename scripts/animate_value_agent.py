"""Record and render the certified search-free RL value agent."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bot2048.env import Game2048
from bot2048.value_agent import ValueEnsembleAgent
from scripts.animate import frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="models/rl/final_value_policy.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="animations/final_rl_bot")
    a = p.parse_args()
    agent = ValueEnsembleAgent(a.manifest)
    game = Game2048(a.seed)
    trajectory = [{"board": game.board.tolist(), "score": 0, "move": 0}]
    while not game.done:
        action = agent.act(game.board)
        transition = game.step(action)
        if not transition.changed:
            raise RuntimeError("value agent selected an illegal action")
        trajectory.append({
            "board": game.board.tolist(), "score": game.score,
            "move": game.moves, "action": action,
        })
    agent.close()

    replay = Game2048(a.seed)
    for state in trajectory[1:]:
        replay.step(state["action"])
        np.testing.assert_array_equal(replay.board, state["board"])
        assert replay.score == state["score"]
    manifest = Path(a.manifest)
    row = {
        "manifest": a.manifest,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "seed": a.seed,
        "environment": "standard 2048; two initial tiles; 90% 2 / 10% 4",
        "search_at_inference": False,
        "inference": "one deterministic afterstate per legal action and learned RL neural values",
        "replay_verified": True,
        "won": game.won,
        "score": game.score,
        "moves": game.moves,
        "max_tile": 2 ** int(game.board.max()),
        "trajectory": trajectory,
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    indices = np.unique(
        np.linspace(0, len(trajectory) - 1, min(150, len(trajectory)), dtype=int)
    )
    title = "Certified RL neural bot"
    subtitle = "Learned value ensemble • no expectimax • accelerated replay"
    frames = [frame(trajectory[i], title, subtitle, seed=a.seed) for i in indices]
    final = frame(trajectory[-1], title, subtitle, True, game.won, a.seed)
    frames.append(final)
    frames[0].save(
        output.with_suffix(".gif"), save_all=True, append_images=frames[1:],
        duration=[90] * (len(frames) - 1) + [2200], loop=0, optimize=True,
    )
    final.save(output.with_name(output.name + "_final.png"))
    print(json.dumps({k: v for k, v in row.items() if k != "trajectory"}))


if __name__ == "__main__":
    main()
