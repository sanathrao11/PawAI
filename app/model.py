from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        use_time_frequency: bool = True,
        use_attention: bool = True,
        cnn_filters: tuple[int, int] = (32, 64),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_time_frequency = use_time_frequency
        self.use_attention = use_attention

        if self.use_time_frequency:
            self.cnn2d = nn.Sequential(
                nn.Conv2d(n_channels, cnn_filters[0], kernel_size=(3, 5), padding=(1, 2)),
                nn.BatchNorm2d(cnn_filters[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(cnn_filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            lstm_input_size = cnn_filters[1]
        else:
            self.cnn1d = nn.Sequential(
                nn.Conv1d(n_channels, cnn_filters[0], kernel_size=7, padding=3),
                nn.BatchNorm1d(cnn_filters[0]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=5, padding=2),
                nn.BatchNorm1d(cnn_filters[1]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            )
            lstm_input_size = cnn_filters[1]

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn_score = nn.Linear(lstm_hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, return_attention: bool = False):
        if self.use_time_frequency:
            x = self.cnn2d(x)
            x = x.mean(dim=2)
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.cnn1d(x)
            x = x.permute(0, 2, 1)

        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attention:
            scores = self.attn_score(lstm_out).squeeze(-1)
            attention_weights = torch.softmax(scores, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        else:
            context = torch.cat([h_n[-2], h_n[-1]], dim=1)
            attention_weights = None

        context = self.dropout(context)
        logits = self.classifier(context)

        if return_attention:
            return logits, attention_weights
        return logits


def build_model(
    n_channels: int,
    n_classes: int,
    use_time_frequency: bool = True,
    use_attention: bool = True,
) -> CNNBiLSTM:
    return CNNBiLSTM(
        n_channels=n_channels,
        n_classes=n_classes,
        use_time_frequency=use_time_frequency,
        use_attention=use_attention,
    )


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f'Unsupported checkpoint format at {path}')
    return state


def load_model_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    state_dict = _load_state_dict(path, device)
    model.load_state_dict(state_dict, strict=True)
