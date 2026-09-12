import unittest

import numpy as np
import torch
from torch import nn

from bot2048.deep_rl import ConvPolicy
from scripts.train_conv_rl import ensemble_logits, loss_and_accuracy


class FixedPolicy(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits[: len(x)]


class TrainingLossTests(unittest.TestCase):
    def test_ensemble_logits_match_manual_alignment(self):
        torch.manual_seed(7)
        model = ConvPolicy(width=4, blocks=1, hidden=8)
        boards = np.random.default_rng(7).integers(0, 12, (3, 4, 4))
        expected = []
        from bot2048.deep_rl import encode

        for reflection in (False, True):
            for k in range(4):
                variant = np.rot90(boards, k, axes=(1, 2))
                mapping = (np.arange(4) - k) % 4
                if reflection:
                    variant = variant[:, :, ::-1]
                    mapping = (-mapping) % 4
                expected.append(model(encode(variant.copy()))[:, mapping.tolist()])
        torch.testing.assert_close(
            ensemble_logits(model, boards, "cpu"), torch.stack(expected).mean(0)
        )

    def test_hard_targets_train_teacher_argmax(self):
        boards = np.zeros((2, 4, 4), dtype=np.uint8)
        q = np.array([[9, 2, -1e30, 1], [1, 8, 2, -1e30]], dtype=np.float32)
        model = FixedPolicy([[0, 4, 0, 0], [4, 0, 0, 0]])
        loss, accuracy = loss_and_accuracy(
            model, boards, q, "cpu", 200, hard_targets=True
        )
        self.assertGreater(loss.item(), 3)
        self.assertEqual(accuracy.item(), 0)
        loss.backward()
        self.assertLess(model.logits.grad[0, 0], 0)
        self.assertLess(model.logits.grad[1, 1], 0)

    def test_late_state_weight_emphasizes_late_error(self):
        boards = np.zeros((2, 4, 4), dtype=np.uint8)
        boards[1, 0, 0] = 10
        q = np.array([[4, 0, -1e30, -1e30], [4, 0, -1e30, -1e30]], dtype=np.float32)
        model = FixedPolicy([[0, 4, 0, 0], [0, 8, 0, 0]])
        plain, _ = loss_and_accuracy(model, boards, q, "cpu", 200, hard_targets=True)
        weighted, _ = loss_and_accuracy(
            model, boards, q, "cpu", 200, hard_targets=True, late_state_weight=3
        )
        self.assertGreater(weighted.item(), plain.item())

    def test_margin_penalizes_costly_wrong_action(self):
        boards = np.zeros((2, 4, 4), dtype=np.uint8)
        q = np.array(
            [[10, 9, -1e30, -1e30], [10, -9990, -1e30, -1e30]], dtype=np.float32
        )
        model = FixedPolicy([[0, 2, 0, 0], [0, 2, 0, 0]])
        loss, _ = loss_and_accuracy(
            model, boards, q, "cpu", 200, hard_targets=True, margin_weight=1
        )
        loss.backward()
        self.assertLess(model.logits.grad[1, 0], model.logits.grad[0, 0])
        self.assertGreater(model.logits.grad[1, 1], model.logits.grad[0, 1])


if __name__ == "__main__":
    unittest.main()
