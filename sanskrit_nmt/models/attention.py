"""
=========================================================
Bahdanau Attention
=========================================================
"""

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):

        super().__init__()

        #  encoder outputs
        self.encoder_projection = nn.Linear(
            encoder_hidden_dim,
            decoder_hidden_dim
        )

        #  decoder hidden state
        self.decoder_projection = nn.Linear(
            decoder_hidden_dim,
            decoder_hidden_dim
        )

        # Compute attention score
        self.energy = nn.Linear(
            decoder_hidden_dim,
            1
        )

    def forward(
        self,
        decoder_hidden,
        encoder_outputs
    ):
        """
        Parameters
        ----------
        decoder_hidden
            (batch, decoder_hidden)

        encoder_outputs
            (batch, seq_len, encoder_hidden)
        """

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Expand decoder hidden across sequence length
        decoder_hidden = decoder_hidden.unsqueeze(1)

        decoder_hidden = decoder_hidden.repeat(
            1,
            seq_len,
            1
        )

        # Compute energy
        energy = torch.tanh(

            self.encoder_projection(encoder_outputs)

            +

            self.decoder_projection(decoder_hidden)

        )

        # Compute scores
        scores = self.energy(energy).squeeze(-1)

        # Normalize
        attention_weights = torch.softmax(
            scores,
            dim=1)
        
        

        # Context vector
        context = torch.bmm(

            attention_weights.unsqueeze(1),

            encoder_outputs

        )

        context = context.squeeze(1)

        return context, attention_weights