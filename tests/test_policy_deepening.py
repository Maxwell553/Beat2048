import unittest

import torch

from bot2048.deep_rl import ConvPolicy
from scripts.deepen_policy import deepen


class DeepeningTests(unittest.TestCase):
    def test_additional_blocks_preserve_logits_exactly(self):
        torch.manual_seed(41)
        source = ConvPolicy(width=8, blocks=2, hidden=16)
        target = deepen(source, blocks=4)
        x = torch.randn(20, 16, 4, 4)
        torch.testing.assert_close(source(x), target(x), rtol=0, atol=0)
        self.assertEqual(target.config["blocks"], 4)


if __name__ == "__main__":
    unittest.main()
