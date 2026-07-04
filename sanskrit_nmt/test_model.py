"""
=========================================================
Test Model script
=========================================================
"""


from torch.utils.data import DataLoader
import random

import torch
import torch.nn as nn
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

def build_vocabularies(dataset, preprocessor, tokenizer):
    """
    Build source and target vocabularies.
    """

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    src_sentences = []
    tgt_sentences = []

    print("Building vocabularies...")

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

def test_encoder(encoder, train_loader):
    """
    Test Encoder using one batch.
    """

    print("\n" + "=" * 60)
    print("Testing Encoder")
    print("=" * 60)

    # Get one batch
    batch = next(iter(train_loader))

    src = batch["src"]

    print("\nInput Tensor Shape")
    print(src.shape)

    # Forward pass
    outputs, hidden, cell = encoder(src)

    print("\nEmbedding + BiLSTM Results")
    print("-" * 60)

    print("Encoder Outputs Shape :")
    print(outputs.shape)

    print("\nHidden State Shape :")
    print(hidden.shape)

    print("\nCell State Shape :")
    print(cell.shape)

    print("\nSample Encoder Output (First Sentence, First Word)")
    print(outputs[0, 0, :10])      # First 10 values

    print("\nSample Hidden State")
    print(hidden[0, 0, :10])

    print("\nSample Cell State")
    print(cell[0, 0, :10])

    print("\nEncoder Test Passed Successfully!")
    return 0

def test_attention(encoder, attention, train_loader):
    """
    Test Bahdanau Attention Module.
    """

    print("\n" + "=" * 60)
    print("Testing Bahdanau Attention")
    print("=" * 60)

    # ----------------------------------------
    # Load One Batch
    # ----------------------------------------

    batch = next(iter(train_loader))

    src = batch["src"]

    print("\nInput Shape")
    print(src.shape)

    # ----------------------------------------
    # Encoder Forward
    # ----------------------------------------

    encoder_outputs, hidden, cell = encoder(src)

    print("\nEncoder Output Shape")
    print(encoder_outputs.shape)

    print("\nHidden Shape")
    print(hidden.shape)

    # ----------------------------------------
    # Decoder Hidden
    # ----------------------------------------

    #decoder_hidden = hidden[-1]   without using the bridge layer 
    decoder_hidden = hidden.squeeze(0)

    print("\nDecoder Hidden Shape")
    print(decoder_hidden.shape)

    # ----------------------------------------
    # Attention Forward
    # ----------------------------------------

    context, attention_weights = attention(
        decoder_hidden,
        encoder_outputs
    )

    # ----------------------------------------
    # Results
    # ----------------------------------------

    print("\nContext Vector Shape")
    print(context.shape)

    print("\nAttention Weights Shape")
    print(attention_weights.shape)

    print("\nFirst Sentence Attention Weights")

    print(attention_weights[0])

    print("\nSum of Attention Weights")

    print(attention_weights[0].sum())

    print("\nContext Vector (First 10 Values)")

    print(context[0][:10])

    print("\nAttention Test Passed Successfully!")
    return 0

def test_decoder(
    encoder,
    decoder,
    train_loader
):

    print("\n" + "="*60)
    print("Testing Decoder")
    print("="*60)

    batch = next(iter(train_loader))

    src = batch["src"]
    tgt = batch["tgt"]

    encoder_outputs, hidden, cell = encoder(src)

    # First decoder input = <SOS>
    input_token = tgt[:, 0]

    prediction, hidden, cell, attn = decoder(

        input_token,

        hidden,

        cell,

        encoder_outputs

    )

    print("\nPrediction Shape")
    print(prediction.shape)

    print("\nHidden Shape")
    print(hidden.shape)

    print("\nCell Shape")
    print(cell.shape)

    print("\nAttention Shape")
    print(attn.shape)

    print("\nDecoder Test Passed Successfully!")
    return 0

def test_seq2seq(
    model,
    train_loader,
    device
):

    print("\n" + "="*60)
    print("Testing Seq2Seq")
    print("="*60)

    batch = next(iter(train_loader))

    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)

    outputs = model(
        src,
        tgt,
        teacher_forcing_ratio=0.5
    )

    print("\nSource Shape")
    print(src.shape)

    print("\nTarget Shape")
    print(tgt.shape)

    print("\nOutput Shape")
    print(outputs.shape)

    print("\nVocabulary Size")
    print(outputs.shape[-1])

    print("\nPrediction for First Word")

    print(outputs[0, 1, :10])

    print("\nSeq2Seq Test Passed Successfully!")    
    return 0
    
def main():

    print("=" * 60)
    print(" Sanskrit → English NMT ")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nLoading Dataset...\n")

    train_dataset = SanskritTranslationDataset(
        cfg.TRAIN_SA_PATH,
        cfg.TRAIN_EN_PATH
    )

    # --------------------------------------------------
    # Preprocessing Objects
    # --------------------------------------------------

    preprocessor = Preprocessor()

    tokenizer = Tokenizer()

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    src_vocab, tgt_vocab = build_vocabularies(
        train_dataset,
        preprocessor,
        tokenizer
    )

    print()

    print("Source Vocabulary :", src_vocab.vocab_size)
    print("Target Vocabulary :", tgt_vocab.vocab_size)

    # --------------------------------------------------
    # Collate Function
    # --------------------------------------------------

    collate_fn = CollateFn(
        preprocessor,
        tokenizer,
        src_vocab,
        tgt_vocab
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    
    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------

    encoder = Encoder(
    input_dim=src_vocab.vocab_size,
    embedding_dim=cfg.EMBEDDING_DIM,
    hidden_dim=cfg.HIDDEN_DIM,
    num_layers=cfg.NUM_LAYERS,
    dropout=cfg.DROPOUT,
    bidirectional=cfg.BIDIRECTIONAL
    )
    
    
    test_encoder(
    encoder,
    train_loader
    )
    # --------------------------------------------------
    # Attention
    # --------------------------------------------------

    attention = BahdanauAttention(
    encoder_hidden_dim=cfg.HIDDEN_DIM * 2,   # BiLSTM Output = 1024
    decoder_hidden_dim=cfg.HIDDEN_DIM        # Decoder Hidden = 512
    )
    test_attention(
    encoder,
    attention,
    train_loader
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
    # Test Decoder
    test_decoder(
        encoder,
        decoder,
        train_loader
    )
    device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )

    model = Seq2Seq(
    encoder,
    decoder,
    device
    ).to(device)
    test_seq2seq(
    model,
    train_loader,
    device
    )
    # --------------------------------------------------
    # Inspect One Batch
    # --------------------------------------------------

    batch = next(iter(train_loader))

    print("=" * 60)
    print("Batch Information")
    print("=" * 60)

    print("Source Shape :", batch["src"].shape)
    print("Target Shape :", batch["tgt"].shape)

    print()

    print("First Source Tensor")

    print(batch["src"][0])

    print()

    print("Recovered Sanskrit Sentence")

    print(
        src_vocab.denumericalize(
            batch["src"][0].tolist()
        )
    )

    print()

    print("Recovered English Sentence")

    print(
        tgt_vocab.denumericalize(
            batch["tgt"][0].tolist()
        )
    )

    print()

    print("Everything is working correctly!")


if __name__ == "__main__":
    main()