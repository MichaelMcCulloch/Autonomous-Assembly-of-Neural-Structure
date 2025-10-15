import os
import random
import torch
from collections import deque
from typing import Dict, Any


class PlayerPool:
    def __init__(self, pool_dir: str, max_size: int = 10):
        self.pool_dir = pool_dir
        self.max_size = max_size
        self.players: deque[str] = deque()
        os.makedirs(pool_dir, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        files = sorted(os.listdir(self.pool_dir))
        for f in files:
            if f.endswith(".pth"):
                self.players.append(os.path.join(self.pool_dir, f))

    def add(self, agent_state_dict: Dict[str, Any], generation: int):
        path = os.path.join(self.pool_dir, f"agent_gen_{generation:04d}.pth")
        torch.save(agent_state_dict, path)
        self.players.append(path)
        if len(self.players) > self.max_size:
            self.players.popleft()

        print(
            f"Added agent generation {generation} to pool. Pool size: {len(self.players)}"
        )

    def get_opponent_state_dict(self) -> Dict[str, Any]:
        if not self.players:
            raise ValueError("Player pool is empty. Cannot get an opponent.")
        opponent_path = random.choice(self.players)
        return torch.load(opponent_path)
