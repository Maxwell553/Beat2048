import json
import unittest
from pathlib import Path

import numpy as np

from bot2048.env import Game2048, legal_actions
from bot2048.value_agent import ValueEnsembleAgent


class ValueEnsembleAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ValueEnsembleAgent("models/rl/final_value_policy.json")

    @classmethod
    def tearDownClass(cls):
        cls.agent.close()

    def test_manifest_is_hash_checked_and_search_free(self):
        manifest = json.loads(Path("models/rl/final_value_policy.json").read_text())
        self.assertIs(manifest["search"], False)
        self.assertNotIn("expectimax", manifest["inference"].lower())
        self.assertEqual(manifest["certification"]["wins"], 100)

    def test_standard_reset_has_exactly_two_tiles(self):
        for seed in range(64):
            game = Game2048(seed)
            self.assertEqual(np.count_nonzero(game.board), 2)
            self.assertTrue(set(game.board.flat) <= {0, 1, 2})

    def test_agent_scores_only_legal_moves(self):
        game = Game2048(42)
        legal = legal_actions(game.board)
        values = self.agent.values(game.board)
        self.assertTrue(np.isfinite(values[legal]).all())
        self.assertIn(self.agent.act(game.board), legal)


if __name__ == "__main__":
    unittest.main()
