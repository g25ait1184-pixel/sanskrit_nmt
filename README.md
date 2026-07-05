# Sanskrit → English Neural Machine Translation

A PyTorch implementation of a **Sequence-to-Sequence Neural Machine Translation (NMT)** system for translating **Sanskrit to English** using a **Bidirectional LSTM Encoder**, **Bahdanau Attention**, and an **LSTM Decoder**.

---

## Project Overview

This project implements an end-to-end Sanskrit → English Neural Machine Translation system that includes:

- Data preprocessing
- Vocabulary construction
- Bidirectional LSTM Encoder
- Bahdanau Attention
- LSTM Decoder
- Seq2Seq architecture
- Model training
- Model evaluation
- Greedy and Beam Search inference
- Submission file generation

---

## Repository Structure

```
Sanskrit_NMT/
│
├── Sanskrit_English_NMT.ipynb        # Complete execution notebook
│
├── sanskrit_nmt/
│   ├── models/
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── attention.py
│   │   └── seq2seq.py
│   │
│   ├── data_loader/
│   │   ├── dataset.py
│   │   └── collate.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── preprocessing.py
│   │   ├── tokenizer.py
│   │   ├── vocabulary.py
│   │   └── metrics.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   │
│   ├── data/
│   ├── checkpoints/
│   ├── outputs/
│   └── logs/
│
├── submission.csv
├── Report.pdf
└── README.md
```

---

## Code Organization

The project is organized into two parts.

### 1. Python Source Code (`sanskrit_nmt/`)

The `sanskrit_nmt` directory contains the complete implementation of the Neural Machine Translation system.

It includes:

- Dataset loading
- Data preprocessing
- Tokenization
- Vocabulary generation
- Encoder
- Attention mechanism
- Decoder
- Seq2Seq model
- Training
- Evaluation
- Inference

Each module is implemented separately to improve readability and maintainability.

---

### 2. Execution Notebook (`Sanskrit_English_NMT.ipynb`)

The notebook demonstrates the complete execution of the project.

It includes:

- Environment setup
- Loading datasets
- Model training
- Model evaluation
- Inference
- Translation examples
- Generation of `submission.csv`

The notebook is intended to reproduce the complete workflow of the project.

---

## Model Architecture

```
Sanskrit Sentence
        │
Embedding Layer
        │
Bidirectional LSTM Encoder
        │
Bahdanau Attention
        │
LSTM Decoder
        │
Linear Output Layer
        │
English Translation
```

---

## Dataset

Training Dataset

- train_sa_10000.csv
- train_en_10000.csv

Test Dataset

- test_sa_1000.csv
- test_en_1000.csv

---

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| Embedding Dimension | 300 |
| Hidden Dimension | 512 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Teacher Forcing Ratio | 0.7 |
| Beam Width | 5 |
| Number of Epochs | 20 |

---

## Evaluation Metrics

The model is evaluated using:

- BLEU Score
- BERTScore (F1)
- Token Accuracy
- Inference Time
- Number of Trainable Parameters

---

## Features

- Bidirectional LSTM Encoder
- Bahdanau Attention
- LSTM Decoder
- Greedy Decoding
- Beam Search Decoding
- Model Checkpointing
- BLEU Evaluation
- BERTScore Evaluation
- Submission File Generation

---

## Running the Project

The complete project execution is available in:

```
Sanskrit_English_NMT.ipynb
```

The notebook demonstrates:

1. Training
2. Evaluation
3. Inference
4. Submission generation

---

## Output

The project generates:

- Trained model checkpoints
- Evaluation metrics
- Translation examples
- `submission.csv`

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
