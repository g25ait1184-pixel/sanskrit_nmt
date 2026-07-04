"""
=========================================================
Project Configuration
---------------------------------------------------------
This file stores all configurable parameters used
throughout the project.

Project: Sanskrit → English Neural Machine Translation
=========================================================
"""

import os
import torch


# =========================================================
# Project Directories
# =========================================================

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Output directories
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# =========================================================
# Dataset Paths
# =========================================================

TRAIN_SA_PATH = os.path.join(RAW_DATA_DIR, "train_sa_10000.csv")
TRAIN_EN_PATH = os.path.join(RAW_DATA_DIR, "train_en_10000.csv")

DEV_SA_PATH = os.path.join(RAW_DATA_DIR, "dev_sa_1000.csv")
DEV_EN_PATH = os.path.join(RAW_DATA_DIR, "dev_en_1000.csv")

TEST_SA_PATH = os.path.join(RAW_DATA_DIR, "test_sa_1000.csv")
TEST_EN_PATH = os.path.join(RAW_DATA_DIR, "test_en_1000.csv")

# -----------------------------
# Dataset Column Names
# -----------------------------
SOURCE_ID_COL = "Source_id"
SANSKRIT_COL = "Sentence_sa"
ENGLISH_COL = "Sentence_en"
# =========================================================
# Special Tokens
# =========================================================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


# =========================================================
# Model Hyperparameters
# =========================================================

EMBEDDING_DIM = 300
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.5
EMBEDDING_DIM = 256
BIDIRECTIONAL = True
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
GRAD_CLIP = 1.0


# =========================================================
# Training Hyperparameters
# =========================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001

# =========================================================
# Checkpoints
# =========================================================

CHECKPOINT_DIR = "checkpoints"

BEST_MODEL = "best_model.pt"

# =========================================================
# Device Configuration
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")