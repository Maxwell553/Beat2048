import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bot2048.env import slide, legal_actions
from bot2048.model import PolicyNet, Agent, save_policy, load_policy, encode
from scripts.train import augment


class ModelTests(unittest.TestCase):
    def test_symmetry_action_mapping(self):
        rng = np.random.default_rng(8)
        boards = rng.integers(0, 9, size=(128, 4, 4))
        actions = rng.integers(0, 4, size=128)
        moved = np.array([slide(b, int(a))[0] for b, a in zip(boards, actions)])
        new_boards, new_actions = augment(boards, actions, np.random.default_rng(999))
        new_moved, _ = augment(moved, actions, np.random.default_rng(999))
        for b, a, expected in zip(new_boards, new_actions, new_moved):
            np.testing.assert_array_equal(slide(b, int(a))[0], expected)

    def test_checkpoint_and_masking(self):
        torch.manual_seed(1)
        m = PolicyNet()
        b = np.zeros((4, 4), dtype=int)
        b[0, 0] = 1
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "test.pt"
            save_policy(p, m, trained=False)
            loaded, meta = load_policy(p)
            torch.testing.assert_close(m(encode(b[None])), loaded(encode(b[None])))
            self.assertFalse(meta["trained"])
            for mode in ["random", "neural", "search", "hybrid"]:
                agent = Agent(p, mode, depth=1)
                self.assertIn(agent.act(b), legal_actions(b))

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in PolicyNet().parameters()), 41412)


if __name__ == "__main__":
    unittest.main()
