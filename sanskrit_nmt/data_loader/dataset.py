"""
=========================================================
Dataset Module
---------------------------------------------------------
This module defines the PyTorch Dataset class for the
Sanskrit → English Neural Machine Translation project.

Responsibilities:
1. Read Sanskrit and English CSV files.
2. Validate dataset columns.
3. Merge the datasets using Source ID.
4. Return parallel sentence pairs.
=========================================================
"""

import pandas as pd
from torch.utils.data import Dataset

from utils.config import (
    SOURCE_ID_COL,
    SANSKRIT_COL,
    ENGLISH_COL
)


class SanskritTranslationDataset(Dataset):
    """
    PyTorch Dataset for Sanskrit-English Translation.

    Each sample contains:
        - Source ID
        - Sanskrit sentence
        - English sentence
    """

    def __init__(self, sa_path, en_path):
        """
        Initialize the dataset.

        Parameters
        ----------
        sa_path : str
            Path to Sanskrit CSV.

        en_path : str
            Path to English CSV.
        """

        # -------------------------------------------------
        # Read CSV files
        # -------------------------------------------------
        self.sa_df = pd.read_csv(sa_path)
        self.en_df = pd.read_csv(en_path)

        # -------------------------------------------------
        # Remove unwanted spaces from column names
        # -------------------------------------------------
        self.sa_df.columns = self.sa_df.columns.str.strip()
        self.en_df.columns = self.en_df.columns.str.strip()

        # -------------------------------------------------
        # Validate required columns
        # -------------------------------------------------
        required_sa = {SOURCE_ID_COL, SANSKRIT_COL}
        required_en = {SOURCE_ID_COL, ENGLISH_COL}

        if not required_sa.issubset(self.sa_df.columns):
            raise ValueError(
                f"Sanskrit CSV missing columns.\n"
                f"Expected : {required_sa}\n"
                f"Found    : {list(self.sa_df.columns)}"
            )

        if not required_en.issubset(self.en_df.columns):
            raise ValueError(
                f"English CSV missing columns.\n"
                f"Expected : {required_en}\n"
                f"Found    : {list(self.en_df.columns)}"
            )

        # -------------------------------------------------
        # Merge Sanskrit and English datasets
        # -------------------------------------------------
        self.data = pd.merge(
            self.sa_df,
            self.en_df,
            on=SOURCE_ID_COL,
            how="inner"
        )

        # -------------------------------------------------
        # Reset DataFrame index
        # -------------------------------------------------
        self.data.reset_index(drop=True, inplace=True)

        print("=" * 60)
        print("Dataset Loaded Successfully")
        print(f"Total Sentence Pairs : {len(self.data)}")
        print("=" * 60)

    def __len__(self):
        """
        Returns the total number of sentence pairs.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns one Sanskrit-English sentence pair.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """

        row = self.data.iloc[index]

        sample = {
            "source_id": row[SOURCE_ID_COL],
            "sanskrit": row[SANSKRIT_COL],
            "english": row[ENGLISH_COL]
        }

        return sample

    def show_sample(self, index=0):
        """
        Display a sample from the dataset.
        """

        sample = self[index]

        print("=" * 60)
        print(f"Source ID : {sample['source_id']}")
        print(f"Sanskrit : {sample['sanskrit']}")
        print(f"English  : {sample['english']}")
        print("=" * 60)

    def dataset_info(self):
        """
        Display dataset information.
        """

        print("\n========== Dataset Information ==========")
        print(f"Total Samples : {len(self)}")
        print(f"Columns       : {list(self.data.columns)}")
        print("\nMissing Values")
        print(self.data.isnull().sum())

        print("\nDuplicate Rows :", self.data.duplicated().sum())
        print("=========================================\n")