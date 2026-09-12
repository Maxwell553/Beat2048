"""Train a compact neural afterstate value model from learned RL targets."""

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np
import torch

from bot2048.deep_rl import AfterstateValueNet, encode
from scripts.train_conv_rl import read


class MoveBatch:
    def __init__(self):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        self.lib = ctypes.CDLL(str(Path(f"native/rl.{suffix}").resolve()))
        self.lib.rl_move_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.rl_move_batch.restype = None

    def afterstates(self, boards):
        boards = np.ascontiguousarray(boards, dtype=np.uint64)
        count = len(boards)
        after = np.empty((count, 4), dtype=np.uint64)
        rewards = np.empty((count, 4), dtype=np.int32)
        for action in range(4):
            self.lib.rl_move_batch(
                boards.ctypes.data,
                count,
                action,
                after[:, action].copy().ctypes.data,
                rewards[:, action].copy().ctypes.data,
            )
        return after, rewards


def unpack(packed):
    return (
        (packed[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 15
    ).astype(np.uint8).reshape(-1, 4, 4)


def augment(boards, rng):
    result = boards.copy()
    choices = rng.integers(0, 8, len(result))
    for transform in range(8):
        selected = choices == transform
        if not selected.any():
            continue
        result[selected] = np.rot90(result[selected], transform % 4, axes=(1, 2))
        if transform >= 4:
            result[selected] = result[selected, :, ::-1]
    return result


def make_afterstates(moves, packed, targets):
    count = len(packed)
    after = np.empty((count, 4), dtype=np.uint64)
    rewards = np.empty((count, 4), dtype=np.int32)
    temporary_after = np.empty(count, dtype=np.uint64)
    temporary_rewards = np.empty(count, dtype=np.int32)
    for action in range(4):
        moves.lib.rl_move_batch(
            packed.ctypes.data,
            count,
            action,
            temporary_after.ctypes.data,
            temporary_rewards.ctypes.data,
        )
        after[:, action] = temporary_after
        rewards[:, action] = temporary_rewards
    legal = targets > -1e20
    if not np.array_equal(legal, after != packed[:, None]):
        raise RuntimeError("native move legality differs from teacher targets")
    return after, rewards, legal


def evaluate(model, moves, boards, targets, batch_size, device, value_scale):
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for offset in range(0, len(boards), batch_size):
            packed = np.ascontiguousarray(boards[offset : offset + batch_size])
            q = targets[offset : offset + len(packed)]
            after, rewards, legal = make_afterstates(moves, packed, q)
            values = model(encode(unpack(after.reshape(-1)), device)).reshape(-1, 4)
            scores = values + torch.as_tensor(rewards, device=device) / value_scale
            mask = torch.as_tensor(legal, device=device)
            scores = scores.masked_fill(~mask, -1e9)
            labels = torch.as_tensor(q.argmax(1), device=device)
            losses.append(torch.nn.functional.cross_entropy(scores, labels).item() * len(packed))
            correct += int((scores.argmax(1) == labels).sum())
            total += len(packed)
    return sum(losses) / total, correct / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--states", type=int, default=0, help="Limit sampled training boards; zero uses all")
    p.add_argument(
        "--validation-states", type=int, default=200000,
        help="Limit validation boards for faster checkpoint feedback; zero uses all",
    )
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--value-scale", type=float, default=1000.0)
    p.add_argument("--value-loss", type=float, default=0.02)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--resume")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train_boards, train_targets = read(a.train)
    validation_boards, validation_targets = read(a.validation)
    train_packed = np.sum(
        train_boards.reshape(-1, 16).astype(np.uint64)
        << (4 * np.arange(16, dtype=np.uint64)), axis=1, dtype=np.uint64
    )
    validation_packed = np.sum(
        validation_boards.reshape(-1, 16).astype(np.uint64)
        << (4 * np.arange(16, dtype=np.uint64)), axis=1, dtype=np.uint64
    )
    if a.states and a.states < len(train_packed):
        selected = rng.choice(len(train_packed), a.states, replace=False)
        train_packed, train_targets = train_packed[selected], train_targets[selected]
    if a.validation_states and a.validation_states < len(validation_packed):
        selected = rng.choice(len(validation_packed), a.validation_states, replace=False)
        validation_packed = validation_packed[selected]
        validation_targets = validation_targets[selected]

    model = AfterstateValueNet(a.width, a.blocks, a.hidden).to(a.device)
    if a.resume:
        artifact = torch.load(a.resume, map_location=a.device, weights_only=True)
        model.load_state_dict(artifact["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    moves = MoveBatch()
    order = np.arange(len(train_packed))
    metadata = {
        "network_config": model.config,
        "value_scale": a.value_scale,
        "training": vars(a),
        "inference": "score each legal deterministic afterstate once; no tree or chance-node search",
    }
    for epoch in range(1, a.epochs + 1):
        rng.shuffle(order)
        model.train()
        total_loss = total_correct = total = 0
        for offset in range(0, len(order), a.batch_size):
            ids = order[offset : offset + a.batch_size]
            packed = np.ascontiguousarray(train_packed[ids])
            q = train_targets[ids]
            after, rewards, legal = make_afterstates(moves, packed, q)
            flat_after = augment(unpack(after.reshape(-1)), rng)
            values = model(encode(flat_after, a.device)).reshape(-1, 4)
            reward_tensor = torch.as_tensor(rewards, device=a.device) / a.value_scale
            scores = values + reward_tensor
            mask = torch.as_tensor(legal, device=a.device)
            masked = scores.masked_fill(~mask, -1e9)
            labels = torch.as_tensor(q.argmax(1), device=a.device)
            ranking = torch.nn.functional.cross_entropy(masked, labels)
            target_values = torch.as_tensor(
                ((q - rewards) / a.value_scale).astype(np.float32), device=a.device
            )
            value_loss = torch.nn.functional.smooth_l1_loss(
                values[mask], target_values[mask]
            )
            loss = ranking + a.value_loss * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ids)
            total_correct += int((masked.argmax(1) == labels).sum())
            total += len(ids)
        validation_loss, validation_accuracy = evaluate(
            model, moves, validation_packed, validation_targets,
            a.batch_size, a.device, a.value_scale,
        )
        artifact = {"state_dict": model.state_dict(), "metadata": metadata}
        torch.save(artifact, a.output)
        epoch_path = Path(a.output).with_name(f"{Path(a.output).stem}.epoch{epoch}.pt")
        torch.save(artifact, epoch_path)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_accuracy": total_correct / total,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }), flush=True)


if __name__ == "__main__":
    main()
