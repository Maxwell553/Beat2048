import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from bot2048.deep_rl import (
    ConvAgent,
    ConvPolicy,
    EquivariantConvPolicy,
    ResidualSymmetryHead,
    encode,
)
from bot2048.env import slide


class DirectPolicyTests(unittest.TestCase):
    def test_equivariant_policy_rotates_action_coordinates(self):
        torch.manual_seed(9)
        model = EquivariantConvPolicy(width=4, blocks=1, hidden=8)
        board = np.array(
            [[1, 0, 2, 0], [0, 3, 0, 1], [2, 0, 4, 0], [0, 1, 0, 0]]
        )
        with torch.no_grad():
            original = model(encode(board[None]))[0]
            rotated = model(encode(np.rot90(board, 1).copy()[None]))[0]
        mapping = torch.tensor([3, 0, 1, 2])
        torch.testing.assert_close(rotated[mapping], original, rtol=1e-5, atol=1e-5)

    def test_untrained_residual_head_exactly_preserves_vote_scores(self):
        torch.manual_seed(1)
        head = ResidualSymmetryHead(hidden=16, scale=4)
        views = torch.randn(3, 8, 4)
        boards = torch.randn(3, 16, 4, 4)
        legal = torch.tensor(
            [[True, True, False, True], [True] * 4, [False, True, True, False]]
        )
        torch.testing.assert_close(
            head(views, boards, legal), head.vote_scores(views, legal)
        )

    def test_one_forward_on_current_board_and_only_legality_checks(self):
        torch.set_num_threads(1)
        board = np.array([[1, 0, 2, 0], [0, 3, 0, 1], [0, 0, 2, 0], [0, 0, 0, 0]])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "network.pt"
            torch.save(
                {
                    "state_dict": ConvPolicy().state_dict(),
                    "metadata": {"trained": False},
                },
                checkpoint,
            )
            with patch(
                "ctypes.CDLL", side_effect=AssertionError("No native engine allowed")
            ):
                agent = ConvAgent(checkpoint)
                inputs = []
                hook = agent.model.register_forward_pre_hook(
                    lambda module, args: inputs.append(args[0].clone())
                )
                with patch("bot2048.env.slide", wraps=slide) as legality:
                    action = agent.act(board)
                hook.remove()
            self.assertEqual(legality.call_count, 4)
            self.assertEqual(len(inputs), 1)
            torch.testing.assert_close(inputs[0], encode(board[None]))
            self.assertIn(action, range(4))

    def test_symmetry_ensemble_only_sees_transforms_of_current_board(self):
        board = np.array([[1, 0, 2, 0], [0, 3, 0, 1], [0, 0, 2, 0], [0, 0, 0, 0]])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "network.pt"
            torch.save(
                {
                    "state_dict": ConvPolicy().state_dict(),
                    "metadata": {"trained": False},
                },
                checkpoint,
            )
            for voting in [False, True]:
                agent = ConvAgent(checkpoint, ensemble=True, voting=voting)
                seen = []
                hook = agent.model.register_forward_pre_hook(
                    lambda module, args: seen.append(args[0].clone())
                )
                with patch(
                    "ctypes.CDLL",
                    side_effect=AssertionError("No native engine allowed"),
                ), patch("bot2048.env.slide", wraps=slide) as legality:
                    agent.act(board)
                hook.remove()
                self.assertEqual(legality.call_count, 4)
                self.assertEqual(len(seen), 1)
                variants = []
                for reflection in (False, True):
                    for k in range(4):
                        b = np.rot90(board, k)
                        if reflection:
                            b = b[:, ::-1]
                        variants.append(b.copy())
                torch.testing.assert_close(seen[0], encode(np.array(variants)))

    def test_neural_committee_only_sees_current_board_symmetries(self):
        board = np.array([[1, 0, 2, 0], [0, 3, 0, 1], [0, 0, 2, 0], [0, 0, 0, 0]])
        with tempfile.TemporaryDirectory() as folder:
            checkpoints = [Path(folder) / f"network-{index}.pt" for index in range(2)]
            for checkpoint in checkpoints:
                torch.save(
                    {
                        "state_dict": ConvPolicy().state_dict(),
                        "metadata": {"trained": False},
                    },
                    checkpoint,
                )
            agent = ConvAgent(
                checkpoints[0],
                ensemble=True,
                voting=True,
                extra_checkpoints=checkpoints[1:],
            )
            seen = [[] for _ in agent.models]
            hooks = [
                model.register_forward_pre_hook(
                    lambda module, args, index=index: seen[index].append(args[0].clone())
                )
                for index, model in enumerate(agent.models)
            ]
            with patch(
                "ctypes.CDLL", side_effect=AssertionError("No native engine allowed")
            ):
                agent.act(board)
            for hook in hooks:
                hook.remove()
            self.assertEqual([len(calls) for calls in seen], [1, 1])
            torch.testing.assert_close(seen[0][0], seen[1][0])


if __name__ == "__main__":
    unittest.main()
