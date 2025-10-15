import numpy as np
from typing import List, Tuple, Optional


class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1

    def reset(self) -> np.ndarray:
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1
        return self.board.flatten()

    def get_state(self) -> np.ndarray:

        return self.board.flatten() * self.current_player

    def get_legal_actions(self) -> List[int]:
        return [i for i, x in enumerate(self.board.flatten()) if x == 0]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.board.flatten()[action] != 0:
            raise ValueError("Illegal move attempted")

        row, col = action // 3, action % 3
        self.board[row, col] = self.current_player

        winner = self._check_winner()
        done = winner is not None or len(self.get_legal_actions()) == 0

        reward = 0.0
        if done:
            if winner is not None:

                reward = 1.0 if winner == self.current_player else -1.0

        self.current_player *= -1

        next_state = self.get_state()

        return next_state, reward, done

    def _check_winner(self) -> Optional[int]:

        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return self.board[i, 0]
            if abs(self.board[:, i].sum()) == 3:
                return self.board[0, i]

        if abs(np.diag(self.board).sum()) == 3:
            return self.board[0, 0]
        if abs(np.diag(np.fliplr(self.board)).sum()) == 3:
            return self.board[0, 2]

        return None

    def render(self):
        chars = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join(chars[x] for x in row))
        print("-" * 5)
