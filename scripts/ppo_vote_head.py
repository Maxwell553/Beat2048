"""PPO-train a residual neural vote head while freezing the strong base policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from bot2048.deep_rl import ConvPolicy, ResidualSymmetryHead, encode
from scripts.ppo_rl import BatchEnv


class VoteActorCritic(nn.Module):
    def __init__(self, checkpoint, hidden=256, scale=2.0):
        super().__init__()
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.base = ConvPolicy(**artifact["metadata"].get("network_config", {}))
        self.base.load_state_dict(artifact["state_dict"])
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.head = ResidualSymmetryHead(hidden, scale)
        self.critic = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.base_artifact = artifact

    def aligned(self, boards):
        variants, mappings = [], []
        for reflection in (False, True):
            for k in range(4):
                variant = torch.rot90(boards, k, dims=(-2, -1))
                mapping = (np.arange(4) - k) % 4
                if reflection:
                    variant = variant.flip(-1)
                    mapping = (-mapping) % 4
                variants.append(variant)
                mappings.append(mapping.tolist())
        with torch.no_grad():
            raw = self.base(torch.cat(variants)).reshape(8, len(boards), 4)
        return torch.stack([raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)

    def forward(self, boards, legal):
        views = self.aligned(boards)
        logits = self.head(views, boards, legal)
        return logits, self.critic(boards).squeeze(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--rollout", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--actor-learning-rate", type=float, default=0.000003)
    p.add_argument("--critic-learning-rate", type=float, default=0.0003)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--win-bonus", type=float, default=20.0)
    p.add_argument("--loss-penalty", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=88000000)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = BatchEnv(args.envs, args.seed)
    model = VoteActorCritic(args.checkpoint, scale=args.scale).to(args.device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.head.parameters(), "lr": args.actor_learning_rate},
            {"params": model.critic.parameters(), "lr": args.critic_learning_rate},
        ],
        eps=1e-5,
    )
    started = time.time()
    for iteration in range(1, args.iterations + 1):
        batches = []
        wins = losses = 0
        model.eval()
        for _ in range(args.rollout):
            boards = env.boards.copy()
            masks = env.legal.copy()
            x = encode(boards, args.device)
            legal = torch.as_tensor(masks, dtype=torch.bool, device=args.device)
            with torch.no_grad():
                logits, values = model(x, legal)
                distribution = torch.distributions.Categorical(
                    logits=(logits.masked_fill(~legal, -1e9) / args.temperature).cpu()
                )
                actions = distribution.sample()
                logp = distribution.log_prob(actions)
            env.step(actions.numpy())
            rewards = env.rewards.copy()
            rewards += (env.outcomes == 1) * (args.win_bonus - 10)
            rewards += (env.outcomes == -1) * (-args.loss_penalty + 1)
            wins += int((env.outcomes == 1).sum())
            losses += int((env.outcomes == -1).sum())
            batches.append(
                (boards, masks, actions.numpy(), logp.numpy(), values.cpu().numpy(), rewards, env.done.copy())
            )
        with torch.no_grad():
            x = encode(env.boards, args.device)
            legal = torch.as_tensor(env.legal, dtype=torch.bool, device=args.device)
            _, bootstrap = model(x, legal)
            next_value = bootstrap.cpu().numpy()
        advantage = np.zeros(args.envs, dtype=np.float32)
        advantages = []
        for row in reversed(batches):
            value, reward, done = row[4], row[5], row[6]
            delta = reward + args.gamma * next_value * (1 - done) - value
            advantage = delta + args.gamma * args.gae_lambda * (1 - done) * advantage
            advantages.append(advantage.copy())
            next_value = value
        advantages = np.stack(advantages[::-1]).reshape(-1)
        values = np.stack([row[4] for row in batches]).reshape(-1)
        returns = advantages + values
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        boards = np.concatenate([row[0] for row in batches])
        masks = np.concatenate([row[1] for row in batches])
        actions = np.concatenate([row[2] for row in batches])
        old_logp = np.concatenate([row[3] for row in batches])
        order = np.arange(len(actions))
        metrics = []
        model.train()
        for _ in range(args.epochs):
            rng.shuffle(order)
            for offset in range(0, len(order), args.batch_size):
                ids = order[offset : offset + args.batch_size]
                x = encode(boards[ids], args.device)
                legal = torch.as_tensor(masks[ids], dtype=torch.bool, device=args.device)
                logits, value = model(x, legal)
                distribution = torch.distributions.Categorical(
                    logits=logits.masked_fill(~legal, -1e9) / args.temperature
                )
                selected = torch.as_tensor(actions[ids], device=args.device)
                logp = distribution.log_prob(selected)
                old = torch.as_tensor(old_logp[ids], device=args.device)
                adv = torch.as_tensor(normalized[ids], device=args.device)
                ratio = (logp - old).exp()
                actor_loss = -torch.minimum(
                    ratio * adv, ratio.clamp(0.9, 1.1) * adv
                ).mean()
                critic_loss = nn.functional.smooth_l1_loss(
                    value, torch.as_tensor(returns[ids], device=args.device)
                )
                entropy = distribution.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), 0.25)
                nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
                optimizer.step()
                metrics.append((actor_loss.item(), critic_loss.item(), entropy.item()))
        row = {
            "iteration": iteration,
            "steps": iteration * args.envs * args.rollout,
            "wins": wins,
            "losses": losses,
            "actor_loss": float(np.mean([m[0] for m in metrics])),
            "critic_loss": float(np.mean([m[1] for m in metrics])),
            "entropy": float(np.mean([m[2] for m in metrics])),
            "seconds": time.time() - started,
        }
        metadata = {
            **model.base_artifact["metadata"],
            "symmetry_head": {"kind": "residual_vote", "hidden": 256, "scale": args.scale},
            "training_method": "PPO on complete standard games with frozen base neural policy",
            "search_at_inference": False,
            "head_training": vars(args),
            **row,
        }
        artifact = {
            "state_dict": model.base_artifact["state_dict"],
            "head_state_dict": {k: v.cpu() for k, v in model.head.state_dict().items()},
            "metadata": metadata,
        }
        destination = Path(args.output).with_name(f"{Path(args.output).stem}.iteration{iteration}.pt")
        torch.save(artifact, str(destination) + ".tmp")
        Path(str(destination) + ".tmp").replace(destination)
        torch.save(artifact, str(args.output) + ".tmp")
        Path(str(args.output) + ".tmp").replace(args.output)
        print(json.dumps(row), flush=True)
    env.close()


if __name__ == "__main__":
    main()
