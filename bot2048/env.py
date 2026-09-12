"""Standard 4x4 2048. Boards contain exponents: 0=empty, 1=2, 11=2048."""

from dataclasses import dataclass
import numpy as np

ACTIONS = ("up", "right", "down", "left")


def slide(board, action):
    """Pure deterministic move; each tile merges at most once. No random spawn."""
    if action not in range(4):
        raise ValueError("action must be 0..3")
    board = np.asarray(board, dtype=np.int16).reshape(4, 4)
    # Rotate so every action can be implemented as a left slide.
    turns = (1, 2, 3, 0)[action]
    work = np.rot90(board, turns)
    result = np.zeros((4, 4), dtype=np.int16)
    reward = 0
    for r in range(4):
        values = [int(v) for v in work[r] if v]
        i = j = 0
        while i < len(values):
            value = values[i]
            if i + 1 < len(values) and value == values[i + 1]:
                value += 1
                reward += 1 << value
                i += 1
            result[r, j] = value
            i += 1
            j += 1
    result = np.rot90(result, -turns).copy()
    return result, reward, not np.array_equal(board, result)


def legal_actions(board):
    return [a for a in range(4) if slide(board, a)[2]]


def pack(board):
    flat = np.asarray(board).ravel()
    if len(flat) != 16 or np.any(flat < 0) or np.any(flat > 14):
        raise ValueError("native planner accepts 16 exponent cells in [0,14]")
    return sum(int(v) << (4 * i) for i, v in enumerate(flat))


def unpack(value):
    return np.array(
        [(value >> (4 * i)) & 15 for i in range(16)], dtype=np.int16
    ).reshape(4, 4)


@dataclass
class Transition:
    board: np.ndarray
    reward: int
    changed: bool
    won: bool
    done: bool


class Game2048:
    def __init__(self, seed=0, board=None, target=2048):
        if target < 2 or target & (target - 1):
            raise ValueError("target must be a power of two")
        self.target_rank = target.bit_length() - 1
        self.rng = np.random.default_rng(seed)
        self.score = 0
        self.moves = 0
        if board is None:
            self.board = np.zeros((4, 4), dtype=np.int16)
            self._spawn()
            self._spawn()
        else:
            arr = np.asarray(board)
            if (
                arr.shape != (4, 4)
                or not np.all(np.isfinite(arr))
                or np.any(arr < 0)
                or np.any(arr > 30)
                or np.any(arr != arr.astype(int))
            ):
                raise ValueError(
                    "board must be a 4x4 array of integer exponents in [0,30]"
                )
            self.board = arr.astype(np.int16).copy()

    def _spawn(self):
        empty = np.flatnonzero(self.board.ravel() == 0)
        if not len(empty):
            return
        cell = int(self.rng.choice(empty))
        self.board.ravel()[cell] = 1 if self.rng.random() < 0.9 else 2

    @property
    def won(self):
        return bool(self.board.max() >= self.target_rank)

    @property
    def done(self):
        return self.won or not legal_actions(self.board)

    def step(self, action):
        if self.done:
            return Transition(self.board.copy(), 0, False, self.won, True)
        next_board, reward, changed = slide(self.board, action)
        if changed:
            self.board = next_board
            self.score += reward
            self.moves += 1
            self._spawn()
        return Transition(self.board.copy(), reward, changed, self.won, self.done)
