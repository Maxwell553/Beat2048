"""Small spatial MLP trained to imitate expected-value search decisions."""

from pathlib import Path
import numpy as np
import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.layers(x)


def encode(boards):
    b = torch.as_tensor(np.asarray(boards), dtype=torch.long)
    return torch.nn.functional.one_hot(b.clamp(0, 15), 16).float()


def save_policy(path, model, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "onehot256-128-64-4",
            "metadata": metadata,
        },
        path,
    )


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PolicyNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["metadata"]


class Agent:
    def __init__(
        self,
        checkpoint=None,
        mode="hybrid",
        seed=0,
        depth=4,
        cutoff=0.0001,
        neural_weight=0.02,
        ensemble=True,
    ):
        from .search import Planner

        self.ensemble = ensemble
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.model = None
        if checkpoint:
            self.model, self.metadata = load_policy(checkpoint)
        if mode in ("neural", "hybrid") and self.model is None:
            raise ValueError("checkpoint required")
        if mode not in ("random", "neural", "search", "hybrid"):
            raise ValueError("unknown mode")
        self.planner = Planner(depth, cutoff) if mode in ("search", "hybrid") else None
        self.neural_weight = neural_weight

    @classmethod
    def from_manifest(cls, path, environment_seed=0):
        import json

        path = Path(path)
        config = json.loads(path.read_text())
        if config.get("type") == "uniform_legal_random":
            return cls(mode="random", seed=environment_seed + 314159)
        import hashlib

        checkpoint = path.parent / config["checkpoint"]
        if (
            config.get("checkpoint_sha256")
            and hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != config["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint SHA-256 does not match saved agent manifest")
        return cls(
            checkpoint=path.parent / config["checkpoint"],
            mode=config["mode"],
            depth=config["depth"],
            cutoff=config["cutoff"],
            neural_weight=config["neural_weight"],
            ensemble=config["symmetry_ensemble"],
        )

    def act(self, board):
        from .env import legal_actions, slide

        legal = legal_actions(board)
        if not legal:
            raise ValueError("no legal actions")
        if self.mode == "random":
            return int(self.rng.choice(legal))
        if self.mode in ("search", "hybrid"):
            for action in legal:
                if slide(board, action)[0].max() >= 11:
                    return action
        if self.mode == "search":
            return self.planner.act(board)
        with torch.no_grad():
            if self.ensemble:
                variants = []
                permutations = []
                for reflection in (False, True):
                    for k in range(4):
                        transformed = np.rot90(board, k)
                        mapping = (np.arange(4) - k) % 4
                        if reflection:
                            transformed = transformed[:, ::-1]
                            mapping = (-mapping) % 4
                        variants.append(transformed.copy())
                        permutations.append(mapping)
                raw = self.model(encode(np.asarray(variants))).numpy()
                logits = np.mean(
                    [raw[i, mapping] for i, mapping in enumerate(permutations)], axis=0
                )
            else:
                logits = self.model(encode(np.asarray(board)[None]))[0].numpy()
        if self.mode == "neural":
            return int(legal[np.argmax(logits[legal])])
        scores = self.planner.scores(board)
        # Neural policy breaks close search decisions; search controls large differences.
        p = torch.softmax(torch.tensor(logits[legal]), dim=0).numpy()
        values = scores[legal]
        scale = max(float(np.ptp(values)), 1.0)
        combined = (values - values.max()) / scale + self.neural_weight * p
        return int(legal[np.argmax(combined)])
