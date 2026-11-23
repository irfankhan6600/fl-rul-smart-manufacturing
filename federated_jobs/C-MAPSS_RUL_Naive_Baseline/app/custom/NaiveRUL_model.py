"""Naive RUL baseline for FL.

The model learns only a single scalar in scaled-target space. After inverse
scaling this is equivalent to a train-set mean RUL baseline. It is intentionally
simple and gives a useful lower-bound comparison for learned temporal models.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


class NaiveRULMeanForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 25,
        input_window: int = 512,
        hidden_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        forecast_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_window = input_window
        self.bias = nn.Parameter(torch.zeros(forecast_size))

    def forward(self, x):
        return self.bias.reshape(1, -1).expand(x.shape[0], -1)


@dataclass
class ModelConfig:
    input_size: int = 25
    input_window: int = 512
    hidden_size: int = 1
    num_layers: int = 1
    dropout: float = 0.0


def build_model(cfg: ModelConfig) -> nn.Module:
    return NaiveRULMeanForecaster(input_size=cfg.input_size, input_window=cfg.input_window)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
