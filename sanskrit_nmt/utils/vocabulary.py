"""
=========================================================
Vocabulary Module
---------------------------------------------------------
Vocabulary class for Sanskrit-English Neural Machine
Translation (NMT).

Responsibilities
----------------
1. Build vocabulary from tokenized sentences.
2. Convert words <-> integer IDs.
3. Handle special tokens.
4. Save and load vocabulary.
=========================================================
"""

import pickle
from collections import Counter

from utils.config import (
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)


class Vocabulary:
    """
    Vocabulary class for one language.

    Example:
        src_vocab = Vocabulary()
        tgt_vocab = Vocabulary()
    """

    def __init__(self, min_freq=1):

        self.min_freq = min_freq

        # Word -> Index
        self.word2idx = {}

        # Index -> Word
        self.idx2word = {}

        # Word Frequency
        self.word_freq = Counter()

        # Initialize special tokens
        self._initialize_special_tokens()

    # =====================================================
    # Initialize Special Tokens
    # =====================================================

    def _initialize_special_tokens(self):

        special_tokens = [
            PAD_TOKEN,
            SOS_TOKEN,
            EOS_TOKEN,
            UNK_TOKEN
        ]

        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    # =====================================================
    # Build Vocabulary
    # =====================================================

    def build_vocab(self, tokenized_sentences):
        """
        Build vocabulary from tokenized sentences.
        """

        # Count word frequencies
        for sentence in tokenized_sentences:
            self.word_freq.update(sentence)

        # Add words satisfying minimum frequency
        for word, freq in self.word_freq.items():

            if freq >= self.min_freq:

                if word not in self.word2idx:

                    idx = len(self.word2idx)

                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

        print("=" * 60)
        print("Vocabulary Built Successfully")
        print(f"Vocabulary Size : {self.vocab_size}")
        print("=" * 60)

    # =====================================================
    # Word -> ID
    # =====================================================

    def lookup_word(self, word):

        return self.word2idx.get(
            word,
            self.word2idx[UNK_TOKEN]
        )

    # =====================================================
    # ID -> Word
    # =====================================================

    def lookup_index(self, index):

        return self.idx2word.get(
            index,
            UNK_TOKEN
        )

    # =====================================================
    # Sentence -> IDs
    # =====================================================

    def numericalize(self, tokens):
        """
        Convert tokens to integer IDs.
        """

        ids = [self.word2idx[SOS_TOKEN]]

        for token in tokens:
            ids.append(self.lookup_word(token))

        ids.append(self.word2idx[EOS_TOKEN])

        return ids

    # =====================================================
    # IDs -> Tokens
    # =====================================================

    def denumericalize(self, ids):
        """
        Convert integer IDs back to tokens.
        """

        tokens = []

        for idx in ids:

            token = self.lookup_index(idx)

            if token in (
                PAD_TOKEN,
                SOS_TOKEN,
                EOS_TOKEN
            ):
                continue

            tokens.append(token)

        return tokens

    # =====================================================
    # Vocabulary Size
    # =====================================================

    @property
    def vocab_size(self):

        return len(self.word2idx)

    # =====================================================
    # Length
    # =====================================================

    def __len__(self):

        return self.vocab_size

    # =====================================================
    # Contains
    # =====================================================

    def __contains__(self, word):

        return word in self.word2idx

    # =====================================================
    # Show Vocabulary
    # =====================================================

    def show_vocab(self, n=20):

        print("=" * 60)

        for i, (word, idx) in enumerate(self.word2idx.items()):

            print(f"{idx:4d} : {word}")

            if i + 1 >= n:
                break

        print("=" * 60)

    # =====================================================
    # Save Vocabulary
    # =====================================================

    def save(self, filepath):

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print(f"Vocabulary saved to {filepath}")

    # =====================================================
    # Load Vocabulary
    # =====================================================

    @staticmethod
    def load(filepath):

        with open(filepath, "rb") as f:
            vocab = pickle.load(f)

        print(f"Vocabulary loaded from {filepath}")

        return vocab