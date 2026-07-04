"""
=========================================================
Tokenizer Module
---------------------------------------------------------
Tokenizer for Sanskrit → English Neural Machine Translation

Sanskrit:
    - Space-based tokenization
    - Handles Devanagari danda (।)

English:
    - Word-level tokenization
    - Separates punctuation

=========================================================
"""

import re


class Tokenizer:

    def __init__(self):
        pass

    # =====================================================
    # Sanskrit Tokenizer
    # =====================================================

    def tokenize_sanskrit(self, sentence: str):
        """
        Tokenize Sanskrit sentence.

        Example
        -------
        Input:
            रामः वनं गच्छति।

        Output:
            ['रामः', 'वनं', 'गच्छति', '।']
        """

        if not isinstance(sentence, str):
            raise TypeError("Sentence must be a string.")

        sentence = sentence.strip()

        # Separate Sanskrit danda
        sentence = sentence.replace("।", " ।")

        # Remove extra spaces
        sentence = re.sub(r"\s+", " ", sentence)

        tokens = sentence.split(" ")

        return tokens

    # =====================================================
    # English Tokenizer
    # =====================================================

    def tokenize_english(self, sentence: str):
        """
        Tokenize English sentence.

        Example
        -------
        Input:
            Rama goes to the forest.

        Output:
            ['Rama','goes','to','the','forest','.']
        """

        if not isinstance(sentence, str):
            raise TypeError("Sentence must be a string.")

        sentence = sentence.strip()

        tokens = re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]",
            sentence
        )

        return tokens

    # =====================================================
    # Generic Tokenizer
    # =====================================================

    def tokenize(self, sentence: str, language="sa"):

        if language.lower() == "sa":
            return self.tokenize_sanskrit(sentence)

        elif language.lower() == "en":
            return self.tokenize_english(sentence)

        else:
            raise ValueError(
                "language must be 'sa' or 'en'"
            )

    # =====================================================
    # Sanskrit Detokenizer
    # =====================================================

    def detokenize_sanskrit(self, tokens):

        sentence = " ".join(tokens)

        sentence = sentence.replace(" ।", "।")

        return sentence

    # =====================================================
    # English Detokenizer
    # =====================================================

    def detokenize_english(self, tokens):

        sentence = " ".join(tokens)

        sentence = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            sentence
        )

        return sentence

    # =====================================================
    # Generic Detokenizer
    # =====================================================

    def detokenize(self, tokens, language="sa"):

        if language.lower() == "sa":
            return self.detokenize_sanskrit(tokens)

        elif language.lower() == "en":
            return self.detokenize_english(tokens)

        else:
            raise ValueError(
                "language must be 'sa' or 'en'"
            )