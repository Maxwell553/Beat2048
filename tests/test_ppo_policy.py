import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvPolicy, encode
from scripts.ppo_rl import ActorCritic


class PPOPolicyTests(unittest.TestCase):
    def test_ensemble_matches_aligned_root_predictions_and_detaches_critic(self):
        torch.set_num_threads(1)
        torch.manual_seed(35)
        policy = ConvPolicy(width=8, blocks=1, hidden=8)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initial.pt"
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "metadata": {"network_config": policy.config},
                },
                checkpoint,
            )
            model = ActorCritic(checkpoint, ensemble=True, detach_critic=True)
        boards = np.random.default_rng(35).integers(0, 12, (3, 4, 4))
        predictions = []
        for reflection in (False, True):
            for k in range(4):
                variant = np.rot90(boards, k, axes=(1, 2))
                mapping = (np.arange(4) - k) % 4
                if reflection:
                    variant = variant[:, :, ::-1]
                    mapping = (-mapping) % 4
                predictions.append(policy(encode(variant.copy()))[:, mapping.tolist()])
        logits, values = model(encode(boards))
        torch.testing.assert_close(logits, torch.stack(predictions).mean(0))
        (values - 1).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.policy.parameters()))
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.critic.parameters()
            )
        )
        model.zero_grad()
        logits, _ = model(encode(boards))
        logits.square().mean().backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.policy.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
