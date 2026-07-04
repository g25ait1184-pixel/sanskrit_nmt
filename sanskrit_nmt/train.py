"""
=========================================================
Train Script

Sanskrit → English Neural Machine Translation
=========================================================
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils.config as cfg

from data_loader.dataset import SanskritTranslationDataset
from data_loader.collate import CollateFn

from utils.preprocessing import Preprocessor
from utils.tokenizer import Tokenizer
from utils.vocabulary import Vocabulary

from models.encoder import Encoder
from models.attention import BahdanauAttention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq


# =========================================================
# Vocabulary
# =========================================================

def build_vocabularies(dataset, preprocessor, tokenizer):
    """
    Build source and target vocabularies.
    """

    print("=" * 60)
    print("Building Vocabulary")
    print("=" * 60)

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    src_sentences = []
    tgt_sentences = []

    for sample in dataset:

        src = preprocessor.preprocess_sanskrit(
            sample["sanskrit"]
        )

        tgt = preprocessor.preprocess_english(
            sample["english"]
        )

        src_tokens = tokenizer.tokenize_sanskrit(src)
        tgt_tokens = tokenizer.tokenize_english(tgt)

        src_sentences.append(src_tokens)
        tgt_sentences.append(tgt_tokens)

    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)

    return src_vocab, tgt_vocab


# =========================================================
# Model
# =========================================================

def create_model(src_vocab, tgt_vocab, device):
    """
    Build complete Seq2Seq model.
    """

    encoder = Encoder(
        input_dim=src_vocab.vocab_size,
        embedding_dim=cfg.EMBEDDING_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        bidirectional=cfg.BIDIRECTIONAL
    )

    attention = BahdanauAttention(
        encoder_hidden_dim=cfg.HIDDEN_DIM * 2,
        decoder_hidden_dim=cfg.HIDDEN_DIM
    )

    decoder = Decoder(
        output_dim=tgt_vocab.vocab_size,
        embedding_dim=cfg.EMBEDDING_DIM,
        encoder_hidden_dim=cfg.HIDDEN_DIM * 2,
        decoder_hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        attention=attention
    )

    model = Seq2Seq(
        encoder,
        decoder,
        device
    ).to(device)

    return model


# =========================================================
# Parameter Count
# =========================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    filename
):

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "loss": loss,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict()

    }

    torch.save(
        checkpoint,
        filename
    )

# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):
    """
    Train model for one epoch.
    """

    # Training mode
    model.train()

    epoch_loss = 0.0

    for batch in train_loader:

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # ---------------------------------------------
        # Clear previous gradients
        # ---------------------------------------------

        optimizer.zero_grad()

        # ---------------------------------------------
        # Forward Pass
        # ---------------------------------------------

        outputs = model(
            src,
            tgt,
            teacher_forcing_ratio=0.7
        )

        # ---------------------------------------------
        # Reshape outputs
        # ---------------------------------------------

        output_dim = outputs.shape[-1]

        outputs = outputs[:, 1:, :].reshape(
            -1,
            output_dim
        )

        tgt = tgt[:, 1:].reshape(-1)

        # ---------------------------------------------
        # Compute Loss
        # ---------------------------------------------

        loss = criterion(
            outputs,
            tgt
        )

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        loss.backward()

        # ---------------------------------------------
        # Gradient Clipping
        # ---------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.GRAD_CLIP
        )

        # ---------------------------------------------
        # Update Parameters
        # ---------------------------------------------

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

# ==========================================================
# Validation
# ==========================================================

def validate_one_epoch(
    model,
    valid_loader,
    criterion,
    device
):
    """
    Validate model.
    """

    model.eval()

    epoch_loss = 0.0

    with torch.no_grad():

        for batch in valid_loader:

            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            outputs = model(
                src,
                tgt,
                teacher_forcing_ratio=0
            )

            output_dim = outputs.shape[-1]

            outputs = outputs[:, 1:, :].reshape(
                -1,
                output_dim
            )

            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(
                outputs,
                tgt
            )

            epoch_loss += loss.item()

    return epoch_loss / len(valid_loader)




# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Sanskrit → English NMT Training")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing Device : {device}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    print("\nLoading Dataset...\n")

    train_dataset = SanskritTranslationDataset(
        cfg.TRAIN_SA_PATH,
        cfg.TRAIN_EN_PATH
    )
    # --------------------------------------------------
    # Validation Dataset
    # --------------------------------------------------

    valid_dataset = SanskritTranslationDataset(
    cfg.DEV_SA_PATH,
    cfg.DEV_EN_PATH
    )


    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    preprocessor = Preprocessor()
    tokenizer = Tokenizer()

    # -------------------------------------------------
    # Vocabulary
    # -------------------------------------------------

    src_vocab, tgt_vocab = build_vocabularies(
        train_dataset,
        preprocessor,
        tokenizer
    )

    print("\nSource Vocabulary :", src_vocab.vocab_size)
    print("Target Vocabulary :", tgt_vocab.vocab_size)

    # -------------------------------------------------
    # DataLoader
    # -------------------------------------------------

    collate_fn = CollateFn(
        preprocessor,
        tokenizer,
        src_vocab,
        tgt_vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
    
    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = create_model(
        src_vocab,
        tgt_vocab,
        device
    )

    print("\nModel Created Successfully")

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.word2idx["<PAD>"]
    )

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE
      )

# -----------------------------------------
# Learning Rate Scheduler
# -----------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer,
          mode="min",
          factor=0.5,
          patience=2
        )

    print("\nTrainable Parameters")
    print(f"{count_parameters(model):,}")
    # -------------------------------------------------
    # Best Validation Loss
    # -------------------------------------------------

    train_losses = []
    valid_losses = []
    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    best_valid_loss = float("inf")
    print("\nStarting Training...\n")
    print(cfg.NUM_EPOCHS)

    for epoch in range(cfg.NUM_EPOCHS):

      start_time = time.time()

      # -----------------------------------------
      # Train
      # -----------------------------------------

      train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
      )

      # -----------------------------------------
      # Validation
      # -----------------------------------------

      valid_loss = validate_one_epoch(
        model,
        valid_loader,
        criterion,
        device
      )
      scheduler.step(valid_loss)
      end_time = time.time()

      epoch_time = end_time - start_time

      print("-" * 60)
      print(f"Epoch : {epoch + 1}/{cfg.NUM_EPOCHS}")
      print(f"Train Loss : {train_loss:.4f}")
      print(f"Valid Loss : {valid_loss:.4f}")
      print(f"Time : {epoch_time:.2f} sec")

      # -----------------------------------------
      # Save Best Model
      # -----------------------------------------

      if valid_loss < best_valid_loss:

          best_valid_loss = valid_loss

          save_checkpoint(
            model,
            optimizer,
            epoch,
            valid_loss,
            os.path.join(
                cfg.CHECKPOINT_DIR,
                cfg.BEST_MODEL
            )
        )

          print(" Best model updated.")

      print("-" * 60)

    print("\nTraining Finished Successfully!")
    print("\nBest Validation Loss : {:.4f}".format(best_valid_loss))


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()