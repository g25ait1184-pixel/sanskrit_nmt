# sanskrit_nmt

# Sanskrit → English Neural Machine Translation (NMT)

## Overview

This project implements a **Neural Machine Translation (NMT)** system for translating **Sanskrit to English** using a **Sequence-to-Sequence (Seq2Seq)** architecture with **Bahdanau Attention**.

The model is implemented in **PyTorch** and includes data preprocessing, vocabulary creation, training, evaluation, beam search inference, and submission file generation.

---

## Features

- Sanskrit text preprocessing
- English text preprocessing
- Custom tokenizer
- Vocabulary generation
- Bidirectional LSTM Encoder
- Bahdanau Attention
- LSTM Decoder
- Greedy Decoding
- Beam Search Decoding
- Training and Validation
- BLEU Score Evaluation
- BERTScore Evaluation
- Token Accuracy
- Model Checkpointing
- Submission CSV Generation

---

## Project Structure

```
sanskrit_nmt/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   ├── seq2seq.py
│   └── transformer.py
│
├── data_loader/
│   ├── dataset.py
│   └── collate.py
│
├── utils/
│   ├── config.py
│   ├── preprocessing.py
│   ├── tokenizer.py
│   ├── vocabulary.py
│   └── metrics.py
│
├── checkpoints/
│
├── outputs/
│
├── logs/
│
├── train.py
├── evaluate.py
├── inference.py
├── README.md
└── requirements.txt
```

---

## Dataset

Training Dataset

- train_sa_10000.csv
- train_en_10000.csv

Validation Dataset

- dev_sa_1000.csv
- dev_en_1000.csv

Test Dataset

- test_sa_1000.csv
- test_en_1000.csv

---

## Model Architecture

```
Sanskrit Sentence
        │
Embedding Layer
        │
BiLSTM Encoder
        │
Bahdanau Attention
        │
LSTM Decoder
        │
Linear Layer
        │
English Translation
```

---

## Hyperparameters

| Parameter | Value |
|-----------|------|
| Embedding Size | 300 |
| Hidden Size | 512 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Teacher Forcing | 0.7 |
| Beam Width | 5 |
| Epochs | 20 |

---

## Installation

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

The best model is automatically saved in:

```
checkpoints/best_model.pt
```

---

## Evaluation

```bash
python evaluate.py
```

Evaluation reports:

- BLEU Score
- BERTScore
- Token Accuracy
- Inference Time
- Number of Parameters

---

## Inference

Generate translations and submission file.

```bash
python inference.py
```

Output:

```
submission.csv
```

---

## Results

Example metrics

| Metric | Value |
|---------|-------|
| BLEU | 0.0137 |
| BERTScore (F1) | 0.8199 |
| Token Accuracy | 0.1232 |
| Parameters | 26.2 Million |

---

## Translation Example

### Sanskrit

```
एक्लिप्स् इति प्रोग्रामर् कृते दोषान्वेषणे अपि साहाय्यं करोति।
```

### Predicted English

```
In this tutorial, we have the...
```

---

## Future Improvements

- Transformer-based NMT
- SentencePiece/BPE Tokenization
- Attention Masking
- Larger Parallel Corpus
- Pre-trained Multilingual Embeddings
- Scheduled Sampling

---

## References

1. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
2. Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014.
3. Papineni et al., *BLEU: A Method for Automatic Evaluation of Machine Translation*, ACL 2002.
4. Zhang et al., *BERTScore*, ICLR 2020.
5. PyTorch Documentation.

---

## Author

**Name:** Vallinath

**email:** g25ait1184@iitj.ac.in

**Course:** Neural Machine Translation

**Framework:** PyTorch

**Language Pair:** Sanskrit → English
