import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bot2048.deep_rl import ConvPolicy, FusionPolicy, encode
from bot2048.env import slide
from bot2048.fusion import FusionAgent


class FakeSparseAgent:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def values_batch(self, boards):
        return np.zeros((len(boards), 4), dtype=np.float32)

    def close(self):
        pass


class FusionAgentTests(unittest.TestCase):
    def test_only_current_board_symmetries_reach_neural_components(self):
        board = np.array(
            [[1, 0, 2, 0], [0, 3, 0, 1], [0, 0, 2, 0], [0, 0, 0, 0]]
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cnn_path, sparse_path, fusion_path = (
                root / "cnn.pt", root / "sparse.bin", root / "fusion.pt"
            )
            torch.save(
                {"state_dict": ConvPolicy().state_dict(), "metadata": {}}, cnn_path
            )
            sparse_path.write_bytes(b"fake sparse checkpoint")
            torch.save(
                {
                    "state_dict": FusionPolicy(300, 16).state_dict(),
                    "metadata": {"inputs": 300, "hidden": 16},
                },
                fusion_path,
            )
            entry = lambda path: {
                "checkpoint": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "mode": "fusion", "cnn": entry(cnn_path),
                "fusion": entry(fusion_path), "sparse": [entry(sparse_path)],
            }))
            with patch("bot2048.fusion.RLAgent", FakeSparseAgent):
                agent = FusionAgent(manifest, device="cpu")
                seen = []
                hook = agent.cnn.model.register_forward_pre_hook(
                    lambda module, args: seen.append(args[0].clone())
                )
                with patch("bot2048.env.slide", wraps=slide) as legality:
                    action = agent.act(board)
                hook.remove()
            self.assertEqual(legality.call_count, 4)
            self.assertEqual(len(seen), 1)
            variants = []
            for reflection in (False, True):
                for k in range(4):
                    transformed = np.rot90(board, k)
                    if reflection:
                        transformed = transformed[:, ::-1]
                    variants.append(transformed.copy())
            torch.testing.assert_close(seen[0], encode(np.asarray(variants)))
            self.assertIn(action, range(4))


if __name__ == "__main__":
    unittest.main()
