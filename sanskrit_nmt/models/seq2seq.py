"""
=========================================================
Seq2Seq Model

Sanskrit -> English Neural Machine Translation

Architecture
------------
Source Sentence -- Encoder (BiLSTM) -- Bridge Layer --
Bahdanau Attention Decoder (LSTM)
--- Vocabulary Prediction
=========================================================
"""

import random
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq Model
    """

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # =====================================================
    # Forward Pass
    # =====================================================

    def forward(
        self,
        src,
        tgt,
        teacher_forcing_ratio=0.5
    ):
        """
        Parameters
        ----------
        src
            Source sentence
            Shape:
                (batch_size, src_len)

        tgt
            Target sentence
            Shape:
                (batch_size, tgt_len)

        teacher_forcing_ratio
            Probability of using ground-truth token.

        Returns
        -------
        outputs
            Shape:
                (batch_size,
                 tgt_len,
                 target_vocab_size)
        """

        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.output_dim

        # Store decoder predictions
        outputs = torch.zeros(
            batch_size,
            tgt_len,
            vocab_size,
            device=self.device
        )

        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src)

        # First decoder input = <SOS>
        input_token = tgt[:, 0]

        # Decoder loop
        for t in range(1, tgt_len):

            prediction, hidden, cell, attention_weights = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:, t, :] = prediction

            top1 = prediction.argmax(dim=1)

            teacher_force = random.random() < teacher_forcing_ratio

            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    # =====================================================
    # Greedy Translation
    # =====================================================

    def translate(
        self,
        src,
        sos_idx,
        eos_idx,
        max_length=50
    ):
        """
        Greedy decoding for inference.
        """

        self.eval()

        with torch.no_grad():

            # Encoder
            encoder_outputs, hidden, cell = self.encoder(src)

            # First decoder input = <SOS>
            input_token = torch.tensor(
                [sos_idx],
                device=self.device
            )

            outputs = []

            for _ in range(max_length):

                prediction, hidden, cell, _ = self.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )

                predicted_token = prediction.argmax(dim=1)
                token = predicted_token.item()

                if token == eos_idx:
                    break

                outputs.append(token)

                input_token = predicted_token

        return outputs

    # =====================================================
    # Beam Search Translation
    # =====================================================

    def beam_search(
        self,
        src,
        sos_idx,
        eos_idx,
        beam_width=5,
        max_length=50
    ):
        """
        Beam Search decoding for inference.
        Works with batch size = 1.
        """

        self.eval()

        with torch.no_grad():

            # -----------------------------------------
            # Encoder
            # -----------------------------------------

            encoder_outputs, hidden, cell = self.encoder(src)

            beams = [
                (
                    [sos_idx],
                    0.0,
                    hidden,
                    cell
                )
            ]

            completed = []

            # -----------------------------------------
            # Decoder
            # -----------------------------------------

            for _ in range(max_length):

                candidates = []

                for seq, score, hidden_state, cell_state in beams:

                    last_token = seq[-1]

                    if last_token == eos_idx:

                        completed.append(
                            (seq, score)
                        )

                        continue

                    input_token = torch.tensor(
                        [last_token],
                        dtype=torch.long,
                        device=self.device
                    )

                    prediction, next_hidden, next_cell, _ = self.decoder(
                        input_token,
                        hidden_state,
                        cell_state,
                        encoder_outputs
                    )

                    log_probs = torch.log_softmax(
                        prediction,
                        dim=1
                    )

                    top_scores, top_indices = torch.topk(
                        log_probs,
                        beam_width,
                        dim=1
                    )

                    for k in range(beam_width):

                        token = top_indices[0, k].item()

                        token_score = top_scores[0, k].item()

                        candidates.append(
                            (
                                seq + [token],
                                score + token_score,
                                next_hidden,
                                next_cell
                            )
                        )

                if len(candidates) == 0:
                    break

                candidates = sorted(
                    candidates,
                    key=lambda x: x[1],
                    reverse=True
                )

                beams = candidates[:beam_width]

            # -----------------------------------------
            # Best Sequence
            # -----------------------------------------

            if len(completed) > 0:

                completed = sorted(
                    completed,
                    key=lambda x: x[1],
                    reverse=True
                )

                best_sequence = completed[0][0]

            else:

                best_sequence = beams[0][0]

            # Remove SOS

            best_sequence = best_sequence[1:]

            # Remove EOS

            if eos_idx in best_sequence:

                eos_position = best_sequence.index(eos_idx)

                best_sequence = best_sequence[:eos_position]

            return best_sequence