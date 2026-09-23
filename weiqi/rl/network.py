"""Small policy/value residual network and explicit accelerator selection."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from ..rl_config import RLTrainingConfig
from .state import INPUT_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs):
        return torch.relu(inputs + self.body(inputs))


class PolicyValueNet(nn.Module):
    def __init__(self, config: RLTrainingConfig) -> None:
        super().__init__()
        net, size = config.network, config.game.board_size
        self.trunk = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, net.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(net.channels), nn.ReLU(),
            *(ResidualBlock(net.channels) for _ in range(net.residual_blocks)),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(net.channels, net.policy_channels, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(net.policy_channels * size * size, size * size + 1),
        )
        self.value = nn.Sequential(
            nn.Conv2d(net.channels, net.value_channels, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(net.value_channels * size * size, net.value_hidden_size),
            nn.ReLU(), nn.Linear(net.value_hidden_size, 1), nn.Tanh(),
        )

    def forward(self, inputs):
        features = self.trunk(inputs)
        return self.policy(features), self.value(features).squeeze(-1)


class Runtime:
    def __init__(self, config: RLTrainingConfig) -> None:
        requested = config.hardware.device
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            torch.cuda.set_device(self.device)
            torch.cuda.set_per_process_memory_fraction(config.hardware.gpu_memory_fraction, self.device)
            torch.backends.cuda.matmul.allow_tf32 = config.hardware.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.hardware.allow_tf32
        elif self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        precision = config.hardware.precision
        # FP32 is the predictable default on older cards, including GTX 1660 Ti.
        self.precision = "float32" if precision == "auto" else precision
        if self.precision != "float32" and self.device.type != "cuda":
            raise ValueError("Mixed precision currently requires CUDA; use float32 on CPU/MPS")
        if self.precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16")
        self.dtype = torch.float16 if self.precision == "amp_float16" else torch.bfloat16
        torch.set_num_threads(2)

    def autocast(self):
        if self.precision == "float32":
            return nullcontext()
        return torch.autocast("cuda", dtype=self.dtype)

    def evaluator(self, model):
        def predict(batch: np.ndarray):
            model.eval()
            with torch.inference_mode(), self.autocast():
                policy, value = model(torch.as_tensor(batch, device=self.device))
            return policy.float().cpu().numpy(), value.float().cpu().numpy()
        return predict
