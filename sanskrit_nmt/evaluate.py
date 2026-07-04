"""
=========================================================
Evaluation Script

Sanskrit → English NMT

Loads trained model and evaluates on test dataset.

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

from utils.metrics import *

from models.encoder import Encoder
from models.attention import BahdanauAttention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

def save_results(loss):

    os.makedirs("outputs", exist_ok=True)

    file_path = os.path.join(
        "outputs",
        "evaluation_results.txt"
    )

    with open(file_path, "w") as f:

        f.write("="*60 + "\n")

        f.write("Evaluation Results\n")

        f.write("="*60 + "\n\n")

        f.write(f"Loss : {loss:.4f}\n")

    print()

    print("Results saved to")

    print(file_path)

def build_vocabularies(dataset, preprocessor, tokenizer):

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    src_sentences = []
    tgt_sentences = []

    print("="*60)
    print("Building Vocabulary")
    print("="*60)

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
    

    
def load_checkpoint(model, checkpoint_path, device):
    """
    Load a trained model checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)
    print("Epoch :", checkpoint["epoch"] + 1)
    print("Validation Loss :", checkpoint["loss"])

    return model

def evaluate(
    model,
    test_loader,
    criterion,
    tgt_vocab,
    device
):
    """
    Evaluate the trained model on the test dataset.
    """

    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    references = []
    hypotheses = []

    # -------------------------------------------------
    # Measure Inference Time
    # -------------------------------------------------

    first_batch = next(iter(test_loader))

    inference_time = measure_inference_time(
        model,
        first_batch["src"],
        first_batch["tgt"],
        device
    )

    with torch.no_grad():

        for batch in test_loader:

            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            outputs = model(
                src,
                tgt,
                teacher_forcing_ratio=0.0
            )

            output_dim = outputs.shape[-1]

            outputs_loss = outputs[:, 1:, :].reshape(
                -1,
                output_dim
            )

            tgt_loss = tgt[:, 1:].reshape(-1)

            loss = criterion(
                outputs_loss,
                tgt_loss
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=2)

            total_accuracy += calculate_accuracy(
                predictions[:, 1:],
                tgt[:, 1:]
            )

            # ------------------------------------------
            # Convert IDs -> Words
            # ------------------------------------------

            for pred, target in zip(predictions, tgt):

                pred_sentence = tgt_vocab.denumericalize(
                    pred.tolist()
                )

                target_sentence = tgt_vocab.denumericalize(
                    target.tolist()
                )

                hypotheses.append(pred_sentence)
                references.append([target_sentence])

    # -------------------------------------------------
    # Compute Metrics
    # -------------------------------------------------

    bleu = calculate_bleu(
        references,
        hypotheses
    )

    ref_strings = [
        " ".join(ref[0])
        for ref in references
    ]

    hyp_strings = [
        " ".join(hyp)
        for hyp in hypotheses
    ]

    bert_precision, bert_recall, bert_f1 = calculate_bertscore(
        ref_strings,
        hyp_strings
    )

    accuracy = total_accuracy / len(test_loader)
    avg_loss = total_loss / len(test_loader)

    parameters = count_parameters(model)

    print_metrics(
        bleu,
        bert_precision,
        bert_recall,
        bert_f1,
        accuracy,
        inference_time,
        parameters
    )

    return avg_loss

def main():

    print("="*60)
    print("Evaluate Sanskrit → English NMT")
    print("="*60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice :", device)

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    test_dataset = SanskritTranslationDataset(
        cfg.TEST_SA_PATH,
        cfg.TEST_EN_PATH
    )

    preprocessor = Preprocessor()

    tokenizer = Tokenizer()

    train_dataset = SanskritTranslationDataset(
        cfg.TRAIN_SA_PATH,
        cfg.TRAIN_EN_PATH
    )

    src_vocab, tgt_vocab = build_vocabularies(
        train_dataset,
        preprocessor,
        tokenizer
    )

    collate_fn = CollateFn(
        preprocessor,
        tokenizer,
        src_vocab,
        tgt_vocab
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = create_model(
        src_vocab,
        tgt_vocab,
        device
    )

    checkpoint_path = os.path.join(
        cfg.CHECKPOINT_DIR,
        cfg.BEST_MODEL
    )

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.word2idx["<PAD>"]
    )

    loss = evaluate(
        model,
        test_loader,
        criterion,
        tgt_vocab,
        device
    )

    print()

    print("Average Loss :", loss)

    save_results(loss)

# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()
