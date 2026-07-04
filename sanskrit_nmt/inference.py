"""
=========================================================
Inference Script

Sanskrit → English Neural Machine Translation

Loads the best trained model and generates submission.csv

=========================================================
"""

import os
import time
import pandas as pd

import torch

import utils.config as cfg

from utils.preprocessing import Preprocessor
from utils.tokenizer import Tokenizer
from utils.vocabulary import Vocabulary

from data_loader.dataset import SanskritTranslationDataset

from models.encoder import Encoder
from models.attention import BahdanauAttention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq


# =========================================================
# Build Vocabulary
# =========================================================

def build_vocabularies(dataset, preprocessor, tokenizer):

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
# Create Model
# =========================================================

def create_model(src_vocab, tgt_vocab, device):

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
# Load Best Model
# =========================================================

def load_checkpoint(model, checkpoint_path, device):

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    print()

    print("=" * 60)

    print("Best Model Loaded")

    print("=" * 60)

    print("Epoch :", checkpoint["epoch"] + 1)

    print("Validation Loss :", checkpoint["loss"])

    return model


# =========================================================
# Count Parameters
# =========================================================

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    # =========================================================
# Translate One Sentence
# =========================================================

def translate_sentence(
    sentence,
    model,
    preprocessor,
    tokenizer,
    src_vocab,
    tgt_vocab,
    device,
    max_length=60
):
    """
    Translate a single Sanskrit sentence into English.
    """

    model.eval()

    # -----------------------------------------
    # Preprocess
    # -----------------------------------------

    sentence = preprocessor.preprocess_sanskrit(sentence)

    # -----------------------------------------
    # Tokenize
    # -----------------------------------------

    tokens = tokenizer.tokenize_sanskrit(sentence)

    # -----------------------------------------
    # Numericalize
    # -----------------------------------------

    ids = src_vocab.numericalize(tokens)

    # Add SOS/EOS if not already added by numericalize()
    if len(ids) == 0 or ids[0] != 1:
        ids = [1] + ids

    if ids[-1] != 2:
        ids.append(2)

    # -----------------------------------------
    # Tensor
    # -----------------------------------------

    src_tensor = torch.tensor(
        ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    # -----------------------------------------
    # Translate
    # -----------------------------------------
    #USE_BEAM_SEARCH = True
    #if USE_BEAM_SEARCH:
    predicted_ids = model.beam_search(
        src_tensor,
        sos_idx=1,
        eos_idx=2,
        beam_width=5,
        max_length=max_length
    )
    #else:
     # predicted_ids = model.translate(
      #  src_tensor,
    #    sos_idx=1,
    #    eos_idx=2,
    #    max_length=max_length
   # )
      

    # -----------------------------------------
    # Convert IDs → Words
    # -----------------------------------------

    predicted_words = tgt_vocab.denumericalize(
        predicted_ids
    )

    # Remove special tokens if present
    predicted_words = [
        word
        for word in predicted_words
        if word not in [
            "<SOS>",
            "<EOS>",
            "<PAD>"
        ]
    ]

    return " ".join(predicted_words)


# =========================================================
# Translate Entire Test Set
# =========================================================

def generate_predictions(
    dataframe,
    model,
    preprocessor,
    tokenizer,
    src_vocab,
    tgt_vocab,
    device
):
    """
    Translate all Sanskrit sentences.
    """

    predictions = []

    start_time = time.time()

    for sentence in dataframe["Sentence_sa"]:

        translation = translate_sentence(
            sentence=sentence,
            model=model,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device
        )

        predictions.append(translation)

    end_time = time.time()

    inference_time = end_time - start_time

    return predictions, inference_time


# =========================================================
# Save Submission
# =========================================================

def save_submission(
    dataframe,
    predictions,
    filename="submission.csv"
):
    """
    Save predictions in assignment format.
    """

    submission = pd.DataFrame({

        "Source_id":
            dataframe["Source_id"],

        "Sentence_en":
            predictions

    })

    submission.to_csv(

        filename,

        index=False,

        encoding="utf-8"

    )

    print()

    print("=" * 60)

    print("submission.csv generated successfully")

    print("=" * 60)

    print("Saved to :", filename)

    # =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Sanskrit → English NMT Inference")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUsing Device :", device)

    # -------------------------------------------------
    # Training Dataset
    # (Used only to rebuild vocabularies)
    # -------------------------------------------------

    train_dataset = SanskritTranslationDataset(

        cfg.TRAIN_SA_PATH,

        cfg.TRAIN_EN_PATH

    )

    preprocessor = Preprocessor()

    tokenizer = Tokenizer()

    # -------------------------------------------------
    # Build Vocabulary
    # -------------------------------------------------

    src_vocab, tgt_vocab = build_vocabularies(

        train_dataset,

        preprocessor,

        tokenizer

    )

    print()

    print("Source Vocabulary :", src_vocab.vocab_size)
    print("Target Vocabulary :", tgt_vocab.vocab_size)

    # -------------------------------------------------
    # Build Model
    # -------------------------------------------------

    model = create_model(

        src_vocab,

        tgt_vocab,

        device

    )

    # -------------------------------------------------
    # Load Best Model
    # -------------------------------------------------

    checkpoint_path = os.path.join(

        cfg.CHECKPOINT_DIR,

        cfg.BEST_MODEL

    )

    model = load_checkpoint(

        model,

        checkpoint_path,

        device

    )

    # -------------------------------------------------
    # Read Test CSV
    # -------------------------------------------------

    print()

    print("Loading Test Dataset...")

    test_df = pd.read_csv(

        cfg.TEST_SA_PATH

    )

    print()

    print("Total Test Sentences :", len(test_df))

    # -------------------------------------------------
    # Translation
    # -------------------------------------------------

    print()

    print("Generating English Translations...")

    predictions, inference_time = generate_predictions(

        dataframe=test_df,

        model=model,

        preprocessor=preprocessor,

        tokenizer=tokenizer,

        src_vocab=src_vocab,

        tgt_vocab=tgt_vocab,

        device=device

    )

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------

    save_submission(

        dataframe=test_df,

        predictions=predictions,

        filename="submission.csv"

    )

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    print()

    print("=" * 60)

    print("Inference Statistics")

    print("=" * 60)

    print("Total Sentences :", len(test_df))

    print("Total Inference Time : {:.4f} sec".format(

        inference_time

    ))

    print("Average Time / Sentence : {:.6f} sec".format(

        inference_time / len(test_df)

    ))

    print("Trainable Parameters : {:,}".format(

        count_parameters(model)

    ))

    print("=" * 60)

    # -------------------------------------------------
    # Show Sample Predictions
    # -------------------------------------------------

    print()

    print("=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    for i in range(min(5, len(test_df))):

        print()

        print("Source ID :", test_df.iloc[i]["Source_id"])

        print("Sanskrit :")

        print(test_df.iloc[i]["Sentence_sa"])

        print()

        print("Prediction :")

        print(predictions[i])

        print("-" * 60)

    print()

    print("Inference Completed Successfully!")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()