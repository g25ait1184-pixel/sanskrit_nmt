"""
=========================================================
Decoder Module

LSTM Decoder with Bahdanau Attention

Author : Your Name
Project: Sanskrit -> English Neural Machine Translation
=========================================================
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention
    """

    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        num_layers,
        dropout,
        attention
    ):
        super().__init__()

        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention = attention

        # -------------------------------------------------
        # Target Embedding
        # -------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------
        # LSTM
        #
        # Input =
        # Embedding + Context Vector
        # -------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # -------------------------------------------------
        # Vocabulary Prediction
        # -------------------------------------------------

        self.fc = nn.Linear(
          decoder_hidden_dim +
          encoder_hidden_dim +
          embedding_dim,
          output_dim
          )

    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_outputs
    ):
        """
        Parameters
        ----------
        input_token : Tensor
            Shape = (batch)

        hidden : Tensor
            Shape = (num_layers, batch, decoder_hidden_dim)

        cell : Tensor
            Shape = (num_layers, batch, decoder_hidden_dim)

        encoder_outputs : Tensor
            Shape = (batch, src_len, encoder_hidden_dim)

        Returns
        -------
        prediction : Tensor
            Shape = (batch, output_dim)

        hidden : Tensor

        cell : Tensor

        attention_weights : Tensor
            Shape = (batch, src_len)
        """

        # -------------------------------------------------
        # Embedding
        # -------------------------------------------------

        embedded = self.embedding(input_token)

        # (batch, embedding_dim)

        embedded = self.dropout(embedded)

        # -------------------------------------------------
        # Attention
        # -------------------------------------------------

        context, attention_weights = self.attention(

            hidden[-1],

            encoder_outputs

        )

        # context
        # (batch, encoder_hidden_dim)

        # -------------------------------------------------
        # Prepare LSTM Input
        # -------------------------------------------------

        lstm_input = torch.cat(

            (

                embedded,

                context

            ),

            dim=1

        )

        # (batch, embedding_dim + encoder_hidden_dim)

        lstm_input = lstm_input.unsqueeze(1)

        # (batch, 1, embedding_dim + encoder_hidden_dim)

        # -------------------------------------------------
        # Decoder LSTM
        # -------------------------------------------------

        output, (hidden, cell) = self.lstm(

            lstm_input,

            (hidden, cell)

        )

        # output
        # (batch,1,decoder_hidden_dim)

        # -------------------------------------------------
        # Vocabulary Prediction
        # -------------------------------------------------

        combined = torch.cat(
            (
                output.squeeze(1),
                context,
                embedded
            ),
            dim=1
        )

        combined = self.dropout(combined)

        prediction = self.fc(combined)



        

        # (batch, output_dim)

        return (

            prediction,

            hidden,

            cell,

            attention_weights

        )