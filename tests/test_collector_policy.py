import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, ConvPolicy
from bot2048.env import Game2048, legal_actions
from scripts.collect_rl_dagger import policy_actions


class CollectorPolicyTests(unittest.TestCase):
    def test_batched_actions_match_deployed_policy(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "policy.pt"
            torch.save(
                {"state_dict": ConvPolicy().state_dict(), "metadata": {"trained": False}},
                checkpoint,
            )
            agent = ConvAgent(checkpoint, ensemble=True)
            games = [Game2048(seed) for seed in range(12)]
            rng = np.random.default_rng(902)
            for _ in range(30):
                for game in games:
                    if not game.done:
                        game.step(int(rng.choice(legal_actions(game.board))))
            boards = np.array([game.board for game in games])
            legal = np.array([[a in legal_actions(b) for a in range(4)] for b in boards])
            for ensemble in (False, True):
                agent.ensemble = ensemble
                expected = [agent.act(board) for board in boards]
                np.testing.assert_array_equal(
                    policy_actions(agent, boards, legal, ensemble), expected
                )
            agent.ensemble = True
            agent.voting = True
            expected = [agent.act(board) for board in boards]
            np.testing.assert_array_equal(
                policy_actions(agent, boards, legal, True, "vote"), expected
            )


if __name__ == "__main__":
    unittest.main()
