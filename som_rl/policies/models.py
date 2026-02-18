from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SOMPolicyBase(nn.Module):
    """
    Minimal base class for SOM policies.

    Subclasses should implement forward(obs) -> (logits, value) where:
      - obs is a dict following the corresponding env's observation_space
      - logits is a (B, action_dim) tensor for a discrete action space
      - value is a (B, 1) tensor with the scalar value estimate
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

    def forward(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BanditMLPPolicy(SOMPolicyBase):
    """
    Bandit archetype policy (for EnvBandit).

    Expects obs dict with:
      - "scalars": (B, 4)
      - "cursor":  (B, 2)
    """

    def __init__(self, action_dim: int, hidden_dim: int = 64):
        super().__init__(action_dim=action_dim)
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        scalars = torch.as_tensor(obs["scalars"], dtype=torch.float32)
        cursor = torch.as_tensor(obs["cursor"], dtype=torch.float32)
        x = torch.cat([scalars, cursor], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class MinesweeperCNNPolicy(SOMPolicyBase):
    """
    Global-grid archetype policy (for EnvMinesweeper).

    Assumes you stack grid channels into obs["grid"] of shape (B, C, H, W) and
    provide:
      - "scalars": (B, 4)
      - "cursor":  (B, 2)
    """

    def __init__(self, action_dim: int, grid_channels: int, hidden_dim: int = 256):
        super().__init__(action_dim=action_dim)
        self.conv1 = nn.Conv2d(grid_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # After the conv stack we will adaptively pool to 5x5 to make the
        # feature size independent of the input H,W.
        self.fc = nn.Linear(64 * 5 * 5 + 64, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = torch.as_tensor(obs["grid"], dtype=torch.float32)  # (B,C,H,W)
        scalars = torch.as_tensor(obs["scalars"], dtype=torch.float32)
        cursor = torch.as_tensor(obs["cursor"], dtype=torch.float32)

        z = F.relu(self.conv1(grid))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.adaptive_avg_pool2d(z, output_size=(5, 5))
        z = torch.flatten(z, start_dim=1)

        s = torch.cat([scalars, cursor], dim=-1)
        s = self.scalar_mlp(s)

        h = torch.cat([z, s], dim=-1)
        h = F.relu(self.fc(h))
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


class ZeldaCNNPolicy(SOMPolicyBase):
    """
    Patch + movement + interact archetype policy (for EnvZelda).

    Assumes obs dict contains:
      - "patch":   (B, C, ps, ps) stacked patch channels
      - "scalars": (B, 4)
      - "cursor":  (B, 2)
    By default this is feedforward; you can wrap it with an RNN externally if
    you want recurrence.
    """

    def __init__(self, action_dim: int, patch_channels: int, hidden_dim: int = 256):
        super().__init__(action_dim=action_dim)
        self.conv1 = nn.Conv2d(patch_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64 * 3 * 3 + 64, hidden_dim)  # 3x3 is a placeholder; adjust once ps is fixed
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = torch.as_tensor(obs["patch"], dtype=torch.float32)  # (B,C,ps,ps)
        scalars = torch.as_tensor(obs["scalars"], dtype=torch.float32)
        cursor = torch.as_tensor(obs["cursor"], dtype=torch.float32)

        z = F.relu(self.conv1(patch))
        z = F.relu(self.conv2(z))
        z = F.adaptive_avg_pool2d(z, output_size=(3, 3))
        z = torch.flatten(z, start_dim=1)

        s = torch.cat([scalars, cursor], dim=-1)
        s = self.scalar_mlp(s)

        h = torch.cat([z, s], dim=-1)
        h = F.relu(self.fc(h))
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

