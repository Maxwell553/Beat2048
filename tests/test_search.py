"""Independent shallow-tree oracle checks the native chance expectation."""

import unittest
import numpy as np
from bot2048.env import slide, legal_actions
from bot2048.search import Planner


def reference_value(board):
    total = 0.0
    for line in list(board) + list(board.T):
        ranks = [int(x) for x in line]
        compact = [x for x in ranks if x]
        runs = []
        for x in compact:
            if runs and runs[-1][0] == x:
                runs[-1][1] += 1
            else:
                runs.append([x, 1])
        merges = sum(n for _, n in runs if n > 1)
        differences = [ranks[i] ** 4 - ranks[i + 1] ** 4 for i in range(3)]
        monotonicity = min(
            sum(max(x, 0) for x in differences), sum(max(-x, 0) for x in differences)
        )
        total += (
            200000
            + 270 * ranks.count(0)
            + 700 * merges
            - 47 * monotonicity
            - 11 * sum(x**3.5 for x in ranks)
        )
    return total


class SearchTests(unittest.TestCase):
    def test_exact_one_step_spawn_expectation(self):
        board = np.array([[0, 1, 3, 4], [2, 3, 5, 7], [3, 4, 0, 8], [2, 2, 1, 9]])
        scores = Planner(depth=1, cutoff=1e-12).scores(board)
        for action in legal_actions(board):
            after, _, _ = slide(board, action)
            empty = np.argwhere(after == 0)
            expected = 0.0
            for r, c in empty:
                for rank, probability in [(1, 0.9), (2, 0.1)]:
                    spawned = after.copy()
                    spawned[r, c] = rank
                    values = [
                        reference_value(slide(spawned, a)[0])
                        for a in legal_actions(spawned)
                    ]
                    expected += (
                        probability * (max(values) if values else 0.0) / len(empty)
                    )
            self.assertAlmostEqual(scores[action], expected, places=7)

    def test_no_legal_action(self):
        board = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])
        with self.assertRaises(ValueError):
            Planner(depth=1).act(board)


if __name__ == "__main__":
    unittest.main()
