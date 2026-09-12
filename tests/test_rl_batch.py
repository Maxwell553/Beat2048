import unittest
import numpy as np
from scripts.ppo_rl import BatchEnv
from bot2048.env import slide, legal_actions


class BatchTests(unittest.TestCase):
    def test_set_boards_updates_state_and_legal_masks(self):
        env = BatchEnv(2, 7)
        try:
            packed = np.array([0x11, 0x22], dtype=np.uint64)
            env.set_boards(packed)
            np.testing.assert_array_equal(env.boards[:, 0, :2], [[1, 1], [2, 2]])
            for board, mask in zip(env.boards, env.legal):
                expected = [int(action in legal_actions(board)) for action in range(4)]
                np.testing.assert_array_equal(mask, expected)
        finally:
            env.close()

    def test_actual_transitions_and_reset(self):
        env = BatchEnv(32, 123456)
        try:
            self.assertTrue(np.all(np.count_nonzero(env.boards, axis=(1, 2)) == 2))
            for _ in range(160):
                before = env.boards.copy()
                actions = np.array(
                    [np.flatnonzero(mask)[0] for mask in env.legal], dtype=np.int32
                )
                env.step(actions)
                for i in range(32):
                    mask = np.array(
                        [int(a in legal_actions(env.boards[i])) for a in range(4)]
                    )
                    np.testing.assert_array_equal(mask, env.legal[i])
                    after, reward, changed = slide(before[i], int(actions[i]))
                    self.assertTrue(changed)
                    expected = reward / 2048 + (
                        10
                        if env.outcomes[i] == 1
                        else -1 if env.outcomes[i] == -1 else 0
                    )
                    self.assertAlmostEqual(env.rewards[i], expected)
                    if env.done[i]:
                        self.assertEqual(np.count_nonzero(env.boards[i]), 2)
                    else:
                        diff = env.boards[i].astype(int) - after
                        self.assertEqual(np.count_nonzero(diff), 1)
                        self.assertIn(diff[diff != 0][0], [1, 2])
                        self.assertEqual(after[diff != 0][0], 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
