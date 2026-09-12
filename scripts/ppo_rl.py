"""PPO fine-tuning with actual rewards, direct neural actions, and no teacher/search."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from bot2048.deep_rl import ConvPolicy, encode


class BatchEnv:
    def __init__(self, n, seed, curriculum=None, curriculum_probability=0):
        root = Path(__file__).resolve().parents[1]
        src = root / "native/rl_batch.cpp"
        lib = src.with_suffix(".dylib" if sys.platform == "darwin" else ".so")
        if (
            not lib.exists()
            or max(src.stat().st_mtime, (root / "native/rl_engine.h").stat().st_mtime)
            > lib.stat().st_mtime
        ):
            temp = lib.with_name(lib.name + ".tmp")
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(src),
                    "-o",
                    str(temp),
                ],
                check=True,
            )
            temp.replace(lib)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.batch_open.argtypes = [ctypes.c_int, ctypes.c_uint64]
        self.lib.batch_open.restype = ctypes.c_void_p
        self.lib.batch_states.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.batch_states.restype = None
        self.lib.batch_set_boards.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.batch_set_boards.restype = None
        self.lib.batch_step.argtypes = [ctypes.c_void_p] * 7
        self.lib.batch_step.restype = None
        self.lib.batch_close.argtypes = [ctypes.c_void_p]
        self.lib.batch_close.restype = None
        self.handle = self.lib.batch_open(n, seed)
        if curriculum is not None:
            self.lib.batch_curriculum.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_double,
            ]
            self.lib.batch_curriculum.restype = None
            curriculum = np.ascontiguousarray(curriculum, dtype=np.uint64)
            self.lib.batch_curriculum(
                self.handle,
                curriculum.ctypes.data,
                len(curriculum),
                curriculum_probability,
            )
        self.boards = np.zeros((n, 4, 4), dtype=np.uint8)
        self.legal = np.zeros((n, 4), dtype=np.uint8)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.done = np.zeros(n, dtype=np.float32)
        self.outcomes = np.zeros(n, dtype=np.int32)
        self.lib.batch_states(
            self.handle, self.boards.ctypes.data, self.legal.ctypes.data
        )

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        self.lib.batch_step(
            self.handle,
            actions.ctypes.data,
            self.rewards.ctypes.data,
            self.done.ctypes.data,
            self.boards.ctypes.data,
            self.legal.ctypes.data,
            self.outcomes.ctypes.data,
        )

    def set_boards(self, packed):
        packed = np.ascontiguousarray(packed, dtype=np.uint64)
        if len(packed) != len(self.boards):
            raise ValueError("one packed board is required per environment")
        self.lib.batch_set_boards(self.handle, packed.ctypes.data)
        self.lib.batch_states(
            self.handle, self.boards.ctypes.data, self.legal.ctypes.data
        )

    def close(self):
        self.lib.batch_close(self.handle)


class ActorCritic(nn.Module):
    def __init__(self, checkpoint, ensemble=False, detach_critic=False):
        super().__init__()
        self.ensemble = ensemble
        self.detach_critic = detach_critic
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.policy = ConvPolicy(**data["metadata"].get("network_config", {}))
        self.policy.load_state_dict(data["state_dict"])
        self.critic = nn.Linear(self.policy.config["hidden"], 1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        if self.ensemble:
            variants, mappings = [], []
            for reflection in (False, True):
                for k in range(4):
                    variant = torch.rot90(x, k, dims=(-2, -1))
                    mapping = (np.arange(4) - k) % 4
                    if reflection:
                        variant = variant.flip(-1)
                        mapping = (-mapping) % 4
                    variants.append(variant)
                    mappings.append(mapping.tolist())
            count = len(x)
            features = self.policy.net[:-1](torch.cat(variants))
            raw = self.policy.net[-1](features).reshape(8, count, 4)
            logits = torch.stack([raw[i][:, m] for i, m in enumerate(mappings)]).mean(0)
            critic_features = features.detach() if self.detach_critic else features
            values = self.critic(critic_features).reshape(8, count).mean(0)
            return logits, values
        features = self.policy.net[:-1](x)
        critic_features = features.detach() if self.detach_critic else features
        return self.policy.net[-1](features), self.critic(critic_features).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/rl/conv_policy.pt")
    p.add_argument("--output", default="models/rl/ppo_policy.pt")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=0.00001)
    p.add_argument("--critic-learning-rate", type=float)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=80000000)
    p.add_argument(
        "--resume", help="Restore optimizer and critic; starts fresh environments"
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--ensemble",
        action="store_true",
        help="Train the deployed symmetry-averaged policy",
    )
    p.add_argument(
        "--detach-critic",
        action="store_true",
        help="Value fitting cannot alter policy features",
    )
    p.add_argument("--entropy-weight", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.99)
    p.add_argument(
        "--curriculum",
        type=float,
        default=0.0,
        help="Probability of resetting to a real expert-game prefix",
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    curriculum = None
    if a.curriculum:
        records = np.fromfile(
            "data/rl_teacher_train.bin",
            dtype=np.dtype([("board", "<u8"), ("q", "<f4", (4,))]),
        )
        ranks = (records["board"][:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 15
        curriculum = records["board"][ranks.max(1) >= 10].copy()
    env = BatchEnv(a.envs, a.seed, curriculum, a.curriculum)
    model = ActorCritic(a.checkpoint, a.ensemble, a.detach_critic).to(a.device)
    parameters = model.parameters()
    if a.critic_learning_rate is not None:
        parameters = [
            {"params": model.policy.parameters(), "lr": a.learning_rate},
            {"params": model.critic.parameters(), "lr": a.critic_learning_rate},
        ]
    optimizer = torch.optim.Adam(parameters, lr=a.learning_rate, eps=1e-5)
    start_iteration = 0
    if a.resume:
        resumed = torch.load(a.resume, map_location=a.device, weights_only=True)
        model.load_state_dict(resumed["actor_critic"])
        optimizer.load_state_dict(resumed["optimizer"])
        for index, group in enumerate(optimizer.param_groups):
            group["lr"] = (
                a.critic_learning_rate
                if index == 1 and a.critic_learning_rate is not None
                else a.learning_rate
            )
        start_iteration = resumed["iteration"]
    gamma = a.gamma
    lam = a.gae_lambda
    started = time.time()
    history = []
    total_wins = total_losses = 0
    for iteration in range(start_iteration + 1, start_iteration + a.iterations + 1):
        model.eval()
        batches = []
        wins = losses = 0
        for t in range(a.rollout):
            b = env.boards.copy()
            mask = env.legal.copy()
            with torch.no_grad():
                logits, value = model(encode(b, a.device))
                logits = logits.masked_fill(
                    ~torch.tensor(mask, dtype=torch.bool, device=a.device), -1e9
                )
                # Sample on CPU to make RNG behavior independent of MPS multinomial support.
                distribution = torch.distributions.Categorical(
                    logits=logits.cpu() / a.temperature
                )
                actions = distribution.sample()
                logp = distribution.log_prob(actions)
            env.step(actions.numpy())
            wins += int((env.outcomes == 1).sum())
            losses += int((env.outcomes == -1).sum())
            batches.append(
                (
                    b,
                    mask,
                    actions.numpy(),
                    logp.numpy(),
                    value.cpu().numpy(),
                    env.rewards.copy(),
                    env.done.copy(),
                )
            )
        with torch.no_grad():
            _, bootstrap = model(encode(env.boards, a.device))
            next_value = bootstrap.cpu().numpy()
        advantage = np.zeros(a.envs, dtype=np.float32)
        advantages = []
        for b, mask, act, logp, value, reward, done in reversed(batches):
            delta = reward + gamma * next_value * (1 - done) - value
            advantage = delta + gamma * lam * (1 - done) * advantage
            advantages.append(advantage.copy())
            next_value = value
        advantage = np.stack(advantages[::-1])
        values = np.stack([r[4] for r in batches])
        returns = (advantage + values).reshape(-1)
        advantage = advantage.reshape(-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        boards = np.concatenate([r[0] for r in batches])
        masks = np.concatenate([r[1] for r in batches])
        actions = np.concatenate([r[2] for r in batches])
        old_logp = np.concatenate([r[3] for r in batches])
        order = np.arange(len(actions))
        model.train()
        metrics = []
        for epoch in range(a.epochs):
            rng.shuffle(order)
            for offset in range(0, len(order), a.batch_size):
                ids = order[offset : offset + a.batch_size]
                logits, value = model(encode(boards[ids], a.device))
                mask = torch.tensor(masks[ids], dtype=torch.bool, device=a.device)
                logits = logits.masked_fill(~mask, -1e9)
                distribution = torch.distributions.Categorical(
                    logits=logits / a.temperature
                )
                act = torch.tensor(actions[ids], device=a.device)
                logp = distribution.log_prob(act)
                old = torch.tensor(old_logp[ids], device=a.device)
                adv = torch.tensor(advantage[ids], device=a.device)
                ratio = (logp - old).exp()
                policy_loss = -torch.minimum(
                    ratio * adv, ratio.clamp(0.8, 1.2) * adv
                ).mean()
                value_loss = nn.functional.smooth_l1_loss(
                    value, torch.tensor(returns[ids], device=a.device)
                )
                entropy = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - a.entropy_weight * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                metrics.append((policy_loss.item(), value_loss.item(), entropy.item()))
        total_wins += wins
        total_losses += losses
        row = {
            "iteration": iteration,
            "steps": iteration * a.envs * a.rollout,
            "wins": wins,
            "losses": losses,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "policy_loss": float(np.mean([m[0] for m in metrics])),
            "value_loss": float(np.mean([m[1] for m in metrics])),
            "entropy": float(np.mean([m[2] for m in metrics])),
            "seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if iteration % 10 == 0 or iteration == start_iteration + a.iterations:
            metadata = {
                "architecture": "configurable-residual-policy4",
                "network_config": model.policy.config,
                "trained": True,
                "method": "PPO on real game rewards after RL-teacher distillation initialization",
                "search_at_inference": False,
                "config": vars(a),
                **row,
            }
            temp = Path(a.output).with_suffix(".tmp")
            torch.save(
                {
                    "state_dict": {
                        k: v.cpu() for k, v in model.policy.state_dict().items()
                    },
                    "metadata": metadata,
                },
                temp,
            )
            temp.replace(a.output)
            # Full optimizer and critic state allow actual continuation, not policy-only restart.
            torch.save(
                {
                    "actor_critic": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "config": vars(a),
                },
                Path(a.output).with_suffix(".training.tmp"),
            )
            Path(a.output).with_suffix(".training.tmp").replace(
                Path(a.output).with_suffix(".training.pt")
            )
            (Path("results/rl") / (Path(a.output).stem + "_training.json")).write_text(
                json.dumps({"config": vars(a), "history": history}, indent=2)
            )
    env.close()


if __name__ == "__main__":
    main()
