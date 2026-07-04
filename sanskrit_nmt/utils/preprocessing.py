"""
=========================================================
Preprocessing Module
---------------------------------------------------------
This module performs basic text preprocessing for the
Sanskrit → English Neural Machine Translation project.

Responsibilities:
1. Normalize Unicode text.
2. Remove extra whitespace.
3. Remove tabs/newlines.
4. Prepare clean text for tokenization.
=========================================================
"""

import re
import unicodedata


class Preprocessor:
    """
    Basic text preprocessor.
    """

    def __init__(self):
        pass

    # =====================================================
    # Common Cleaning
    # =====================================================

    def clean_text(self, sentence: str) -> str:
        """
        Perform common preprocessing steps.

        Parameters
        ----------
        sentence : str

        Returns
        -------
        str
        """

        if not isinstance(sentence, str):
            raise TypeError("Input must be a string.")

        # Unicode normalization
        sentence = unicodedata.normalize("NFC", sentence)

        # Remove tabs/newlines
        sentence = sentence.replace("\n", " ")
        sentence = sentence.replace("\t", " ")

        # Replace multiple spaces with single space
        sentence = re.sub(r"\s+", " ", sentence)

        # Remove leading/trailing spaces
        sentence = sentence.strip()

        return sentence

    # =====================================================
    # Sanskrit
    # =====================================================

    def preprocess_sanskrit(self, sentence: str) -> str:
        """
        Preprocess Sanskrit sentence.
        """

        sentence = self.clean_text(sentence)

        return sentence

    # =====================================================
    # English
    # =====================================================

    def preprocess_english(self, sentence: str) -> str:
        """
        Preprocess English sentence.
        """

        sentence = self.clean_text(sentence)

        return sentence

    # =====================================================
    # Generic
    # =====================================================

    def preprocess(self, sentence: str, language="sa") -> str:
        """
        Generic preprocessing interface.

        Parameters
        ----------
        sentence : str

        language : str
            "sa" -> Sanskrit
            "en" -> English
        """

        if language == "sa":
            return self.preprocess_sanskrit(sentence)

        elif language == "en":
            return self.preprocess_english(sentence)

        else:
            raise ValueError("Language must be 'sa' or 'en'")