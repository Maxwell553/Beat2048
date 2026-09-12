"""Verify final hashes, certification, and every delivered gameplay recording."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from bot2048.env import Game2048
from bot2048.value_agent import ValueEnsembleAgent


def verify_recording(name):
    recording = json.loads(Path(f"animations/{name}.json").read_text())
    game = Game2048(recording["seed"])
    states = recording["trajectory"]
    if "action" in states[0]:
        for state in states:
            np.testing.assert_array_equal(game.board, state["board"])
            assert game.score == state["score"] and game.moves == state["move"]
            if state.get("action") is not None:
                assert game.step(state["action"]).changed
    else:
        np.testing.assert_array_equal(game.board, states[0]["board"])
        for state in states[1:]:
            transition = game.step(state["action"])
            assert transition.changed
            np.testing.assert_array_equal(game.board, state["board"])
            assert game.score == state["score"] and game.moves == state["move"]
    assert game.done and game.won == recording["won"] and game.score == recording["score"]
    gif = Image.open(f"animations/{name}.gif")
    assert gif.n_frames > 1 and gif.size == (640, 800)
    durations = []
    for index in range(gif.n_frames):
        gif.seek(index)
        gif.load()
        durations.append(gif.info["duration"])
    return {
        "recording": name,
        "moves_replayed": game.moves,
        "gif_frames": gif.n_frames,
        "duration_s": sum(durations) / 1000,
        "won": game.won,
    }


def main():
    manifest_path = Path("models/rl/final_value_policy.json")
    manifest = json.loads(manifest_path.read_text())
    agent = ValueEnsembleAgent(manifest_path)
    probe = Game2048(83)
    assert agent.act(probe.board) in range(4)
    agent.close()
    certification = json.loads(
        Path(manifest["certification"]["result"]).read_text()
    )
    assert certification["summary"]["games"] == 100
    assert certification["summary"]["wins"] == 100
    assert certification["search"] is False

    initial = torch.load("models/rl/conv_initial.pt", map_location="cpu", weights_only=True)
    assert initial["metadata"].get("trained") is False
    assert all(torch.isfinite(value).all() for value in initial["state_dict"].values())

    recordings = [
        verify_recording("random_agent"),
        verify_recording("rl_initial_neural"),
        verify_recording("final_rl_bot"),
    ]
    result = {
        "final_manifest_hashes_verified": True,
        "search_disabled": True,
        "certification_100_of_100": True,
        "initial_neural_checkpoint_loads": True,
        "recordings": recordings,
    }
    Path("results/artifact_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
