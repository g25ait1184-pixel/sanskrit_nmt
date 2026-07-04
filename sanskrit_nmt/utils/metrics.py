"""
=========================================================
metrics.py

Evaluation Metrics for Sanskrit → English NMT

Provides:
1. BLEU Score
2. BERTScore
3. Token Accuracy
4. Parameter Count
5. Inference Time
=========================================================
"""

import time
import torch

from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction
)

from bert_score import score


# ==========================================================
# BLEU Score
# ==========================================================

def calculate_bleu(references, hypotheses):
    """
    Calculate Corpus BLEU Score.

    Parameters
    ----------
    references : List[List[List[str]]]
    hypotheses : List[List[str]]

    Returns
    -------
    float
    """

    smoothie = SmoothingFunction().method4

    bleu = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=smoothie
    )

    return bleu


# ==========================================================
# BERTScore
# ==========================================================

def calculate_bertscore(
    references,
    hypotheses,
    lang="en"
):
    """
    Calculate BERTScore.

    Returns
    -------
    precision, recall, f1
    """

    P, R, F1 = score(
        hypotheses,
        references,
        lang=lang,
        verbose=False
    )

    return (
        P.mean().item(),
        R.mean().item(),
        F1.mean().item()
    )


# ==========================================================
# Token Accuracy
# ==========================================================

def calculate_accuracy(
    predictions,
    targets,
    pad_idx=0
):
    """
    Calculate token-level accuracy while ignoring PAD tokens.

    Parameters
    ----------
    predictions : Tensor
    targets : Tensor
    pad_idx : int

    Returns
    -------
    float
    """

    # reshape works for both contiguous and non-contiguous tensors
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    # Ignore PAD tokens
    mask = targets != pad_idx

    predictions = predictions[mask]
    targets = targets[mask]

    if len(targets) == 0:
        return 0.0

    correct = (predictions == targets).sum().item()

    accuracy = correct / len(targets)

    return accuracy


# ==========================================================
# Count Parameters
# ==========================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# Measure Inference Time
# ==========================================================

def measure_inference_time(
    model,
    src,
    tgt,
    device,
    runs=20
):
    """
    Measure average inference time.
    """

    model.eval()

    src = src.to(device)
    tgt = tgt.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(runs):

            model(
                src,
                tgt,
                teacher_forcing_ratio=0
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) / runs


# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(
    bleu,
    bert_precision,
    bert_recall,
    bert_f1,
    accuracy,
    inference_time,
    parameters
):
    """
    Print evaluation metrics.
    """

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"BLEU Score       : {bleu:.4f}")
    print(f"BERT Precision   : {bert_precision:.4f}")
    print(f"BERT Recall      : {bert_recall:.4f}")
    print(f"BERT F1          : {bert_f1:.4f}")
    print(f"Token Accuracy   : {accuracy:.4f}")
    print(f"Inference Time   : {inference_time:.6f} sec")
    print(f"Parameters       : {parameters:,}")

    print("=" * 60)