"""
LSTM Forecasting Model

Input shape:
    (batch_size, window_size, num_features)

Example:
    (256, 96, 25)

Output shape:
    (batch_size, 1)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Sequence-to-one regression model.

    Input:
        (batch, seq_len, features)

    Output:
        (batch, 1)
    """

    def __init__(
        self,
        input_size: int = 25,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        """
        x:
            (batch, seq_len, features)

        returns:
            (batch, 1)
        """

        lstm_out, _ = self.lstm(x)

        # last timestep representation
        x = lstm_out[:, -1, :]

        return self.regressor(x)


@dataclass
class ModelConfig:
    """
    Centralized baseline configuration.
    """

    input_size: int = 25
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


def build_model(cfg: ModelConfig) -> nn.Module:
    return LSTMForecaster(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    cfg = ModelConfig()

    model = build_model(cfg)

    x = torch.randn(32, 96, 25)

    y = model(x)

    print("Output shape:", y.shape)
    print("Trainable parameters:", count_parameters(model))
