import unittest
import torch
from bot2048.deep_rl import ConvPolicy
from scripts.widen_policy import widen


class WideningTests(unittest.TestCase):
    def test_preserves_current_board_logits(self):
        torch.manual_seed(81)
        torch.set_num_threads(1)
        model = ConvPolicy(width=16, hidden=32)
        expanded = widen(model, width=32, hidden=64, noise=0)
        boards = torch.randint(0, 11, (8, 4, 4))
        x = torch.nn.functional.one_hot(boards, 16).permute(0, 3, 1, 2).float()
        torch.testing.assert_close(model(x), expanded(x), rtol=1e-5, atol=1e-6)
        self.assertGreater(
            sum(p.numel() for p in expanded.parameters()),
            sum(p.numel() for p in model.parameters()),
        )


if __name__ == "__main__":
    unittest.main()
