"""
=========================================================
Collate Module
---------------------------------------------------------
Creates mini-batches for the DataLoader.

Pipeline
--------
Batch ,Preprocess,Tokenize,Vocabulary,Padding,Tensor
=========================================================
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.config import PAD_TOKEN


class CollateFn:
    """
    Custom collate function for DataLoader.
    """

    def __init__(
        self,
        preprocessor,
        tokenizer,
        src_vocab,
        tgt_vocab
    ):

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_pad_idx = src_vocab.word2idx[PAD_TOKEN]
        self.tgt_pad_idx = tgt_vocab.word2idx[PAD_TOKEN]

    def __call__(self, batch):
        """
        Parameters
        ----------
        batch : list

            List of samples from Dataset

        Returns
        -------
        dict
        """

        source_ids = []

        src_sequences = []
        tgt_sequences = []

        for sample in batch:

            # -----------------------------------------
            # Source ID
            # -----------------------------------------
            source_ids.append(sample["source_id"])

            # -----------------------------------------
            # Sanskrit
            # -----------------------------------------
            src_sentence = self.preprocessor.preprocess_sanskrit(
                sample["sanskrit"]
            )

            src_tokens = self.tokenizer.tokenize_sanskrit(
                src_sentence
            )

            src_ids = self.src_vocab.numericalize(
                src_tokens
            )

            src_sequences.append(
                torch.tensor(
                    src_ids,
                    dtype=torch.long
                )
            )

            # -----------------------------------------
            # English
            # -----------------------------------------
            tgt_sentence = self.preprocessor.preprocess_english(
                sample["english"]
            )

            tgt_tokens = self.tokenizer.tokenize_english(
                tgt_sentence
            )

            tgt_ids = self.tgt_vocab.numericalize(
                tgt_tokens
            )

            tgt_sequences.append(
                torch.tensor(
                    tgt_ids,
                    dtype=torch.long
                )
            )

        # ---------------------------------------------
        # Padding
        # ---------------------------------------------

        src_batch = pad_sequence(
            src_sequences,
            batch_first=True,
            padding_value=self.src_pad_idx
        )

        tgt_batch = pad_sequence(
            tgt_sequences,
            batch_first=True,
            padding_value=self.tgt_pad_idx
        )

        return {

            "source_id": source_ids,

            "src": src_batch,

            "tgt": tgt_batch

        }