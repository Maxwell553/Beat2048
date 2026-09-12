import ctypes
import unittest
import numpy as np
from bot2048.env import pack, unpack, slide
from bot2048.rl import build


class RLTests(unittest.TestCase):
    def test_rl_engine_matches_reference(self):
        lib = ctypes.CDLL(str(build()))
        lib.rl_move.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.rl_move.restype = ctypes.c_uint64
        rng = np.random.default_rng(89)
        for _ in range(400):
            board = rng.integers(0, 12, size=(4, 4))
            for action in range(4):
                expected, reward, _ = slide(board, action)
                actual_reward = ctypes.c_int()
                actual = unpack(
                    lib.rl_move(pack(board), action, ctypes.byref(actual_reward))
                )
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_reward.value, reward)


if __name__ == "__main__":
    unittest.main()
