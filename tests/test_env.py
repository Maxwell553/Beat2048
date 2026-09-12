import unittest
import numpy as np
from bot2048.env import Game2048, slide, legal_actions, pack, unpack
from bot2048.search import Planner


class EnvironmentTests(unittest.TestCase):
    def test_standard_start_has_two_random_tiles(self):
        for seed in range(100):
            game = Game2048(seed)
            occupied = game.board[game.board != 0]
            self.assertEqual(len(occupied), 2)
            self.assertTrue(np.all(np.isin(occupied, [1, 2])))
            self.assertEqual(game.score, 0)
            self.assertEqual(game.moves, 0)
            self.assertFalse(game.done)

    def test_single_merge_and_reward(self):
        b = np.zeros((4, 4), dtype=int)
        b[0] = [1, 1, 1, 1]
        out, reward, changed = slide(b, 3)
        np.testing.assert_array_equal(out[0], [2, 2, 0, 0])
        self.assertEqual(reward, 8)
        self.assertTrue(changed)
        b[0] = [1, 1, 2, 0]
        np.testing.assert_array_equal(slide(b, 3)[0][0], [2, 2, 0, 0])

    def test_directions(self):
        b = np.zeros((4, 4), dtype=int)
        b[1, 1] = 1
        for action, cell in enumerate([(0, 1), (1, 3), (3, 1), (1, 0)]):
            out, _, _ = slide(b, action)
            self.assertEqual(out[cell], 1)
            self.assertEqual(out.sum(), 1)

    def test_mass_and_native_agreement(self):
        p = Planner()
        rng = np.random.default_rng(44)
        for _ in range(300):
            b = rng.integers(0, 12, size=(4, 4))
            for a in range(4):
                out, _, _ = slide(b, a)
                native = unpack(p.lib.native_move(pack(b), a))
                np.testing.assert_array_equal(out, native)
                self.assertEqual(
                    sum(2 ** int(v) for v in b.ravel() if v),
                    sum(2 ** int(v) for v in out.ravel() if v),
                )

    def test_seed_replay_and_invalid_no_spawn(self):
        b = np.zeros((4, 4), dtype=int)
        b[0, 0] = 1
        g = Game2048(2, b)
        before = g.board.copy()
        self.assertFalse(g.step(0).changed)
        np.testing.assert_array_equal(g.board, before)
        a = Game2048(9)
        b = Game2048(9)
        for _ in range(50):
            action = legal_actions(a.board)[0]
            a.step(action)
            b.step(action)
            np.testing.assert_array_equal(a.board, b.board)

    def test_terminal_start(self):
        b = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])
        g = Game2048(0, b)
        self.assertTrue(g.done)
        self.assertFalse(g.won)
        b[0, 0] = 11
        self.assertTrue(Game2048(0, b).won)

    def test_spawn_distribution(self):
        g = Game2048(10)
        counts = np.zeros(16, dtype=int)
        fours = 0
        for _ in range(10000):
            g.board[:] = 0
            g._spawn()
            cell = np.flatnonzero(g.board.ravel())[0]
            counts[cell] += 1
            fours += g.board.ravel()[cell] == 2
        self.assertTrue(850 < fours < 1150)
        self.assertTrue(np.all((counts > 500) & (counts < 750)))


if __name__ == "__main__":
    unittest.main()
