# src/model.py
# RNN-LSTM architecture used in the PFO-RS framework

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    RNN-LSTM classifier used for crop recommendation.

    Designed to work for:
    - tabular inputs (seq_len = 1)
    - optional sequential inputs (seq_len > 1)
    """

    def __init__(self, input_dim, num_classes, cfg):
        super().__init__()

        lstm_cfg = cfg["model"]["lstm"]
        head_cfg = cfg["model"]["head"]

        hidden_size = lstm_cfg["hidden_size"]
        num_layers = lstm_cfg["num_layers"]
        dropout = lstm_cfg["dropout"]
        bidirectional = lstm_cfg["bidirectional"]

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_dropout = nn.Dropout(lstm_cfg["input_dropout"])

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_size * self.num_directions

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, head_cfg["mlp_hidden"]),
            nn.ReLU(),
            nn.Dropout(head_cfg["mlp_dropout"]),
            nn.Linear(head_cfg["mlp_hidden"], num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Input shape:
        - tabular: (batch, features) → converted to (batch,1,features)
        - sequential: (batch, seq_len, features)
        """

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_dropout(x)

        lstm_out, _ = self.lstm(x)

        last_step = lstm_out[:, -1, :]

        logits = self.classifier(last_step)

        return logits


class YieldRegressor(nn.Module):
    """
    Optional regression head for crop yield prediction.
    """

    def __init__(self, input_dim, cfg):
        super().__init__()

        lstm_cfg = cfg["model"]["lstm"]
        head_cfg = cfg["model"]["head"]

        hidden_size = lstm_cfg["hidden_size"]
        num_layers = lstm_cfg["num_layers"]
        dropout = lstm_cfg["dropout"]
        bidirectional = lstm_cfg["bidirectional"]

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_dropout = nn.Dropout(lstm_cfg["input_dropout"])

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_size * self.num_directions

        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_dim, head_cfg["mlp_hidden"]),
            nn.ReLU(),
            nn.Dropout(head_cfg["mlp_dropout"]),
            nn.Linear(head_cfg["mlp_hidden"], 1),
        )

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_dropout(x)

        lstm_out, _ = self.lstm(x)

        last_step = lstm_out[:, -1, :]

        output = self.regressor(last_step)

        return output.squeeze(-1)
