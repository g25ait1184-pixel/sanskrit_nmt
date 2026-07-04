"""
=========================================================
Encoder Module

Bidirectional LSTM Encoder
for Sanskrit → English Neural Machine Translation

Input
-----
(batch_size, sequence_length)

Output
------
Encoder Outputs
Hidden State
Cell State
=========================================================
"""

import torch
import torch.nn as nn

import utils.config as cfg


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        bidirectional=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # -----------------------------------------
        # Embedding Layer
        # -----------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # -----------------------------------------
        # Dropout
        # -----------------------------------------
        self.dropout = nn.Dropout(dropout)

        # -----------------------------------------
        # BiLSTM
        # -----------------------------------------
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # -----------------------------------------
        # Bridge Layers
        # -----------------------------------------

        bridge_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.hidden_bridge = nn.Linear(
              bridge_input_dim,
              hidden_dim
        )

        self.cell_bridge = nn.Linear(
              bridge_input_dim,
              hidden_dim
          )

    def forward(self, src):
        """
        Parameters
        ----------
        src

        Shape:
            (batch_size, sequence_length)
        """

        # -----------------------------------------
        # Embedding
        # -----------------------------------------
        embedded = self.embedding(src)

        embedded = self.dropout(embedded)

        # -----------------------------------------
        # LSTM
        # -----------------------------------------
        outputs, (hidden, cell) = self.lstm(embedded)

        # -----------------------------------------
        # Bridge Bidirectional States
        # -----------------------------------------

        if self.bidirectional:

            # hidden:
            # (2, batch, hidden)

          hidden = torch.cat(
              (hidden[-2], hidden[-1]),
              dim=1
          )

          hidden = torch.relu(
              self.hidden_bridge(hidden)
          )

          hidden = hidden.unsqueeze(0)

    # -------------------------------------

          cell = torch.cat(
              (cell[-2], cell[-1]),
              dim=1
          )

          cell = torch.relu(
              self.cell_bridge(cell)
          )

          cell = cell.unsqueeze(0)

          return outputs, hidden, cell

        