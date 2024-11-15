# Transformer-based Neural Machine Translation Assignment

This repository contains an implementation of the Transformer architecture for machine translation, based on the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The implementation includes a complete training pipeline for neural machine translation using PyTorch.

## Overview

The codebase consists of two main Python scripts:
- `attention_is_all_you_need.py`: The main implementation file containing the Transformer model and training logic
- `grade.py`: A utility script for evaluating translation quality using BLEU scores

### Features

- Complete Transformer architecture implementation including:
  - Multi-head attention mechanism
  - Positional encoding
  - Layer normalization
  - Feed-forward networks
  - Encoder and decoder stacks
- Batched training with padding
- BLEU score evaluation
- Checkpoint saving and loading
- Attention visualization capabilities
- Support for BPE (Byte Pair Encoding) tokenized input

## Requirements

- Python 3.x
- PyTorch
- NLTK (for BLEU score calculation)
- tqdm (for progress bars)
- matplotlib (for visualization)

## Usage

### Training a Translation Model

```bash
python attention_is_all_you_need.py \
    --src_lang fr \
    --tgt_lang en \
    --train_file data/fren.train.bpe \
    --dev_file data/fren.dev.bpe \
    --test_file data/fren.test.bpe \
    --out_file translations.txt
```

Key Parameters:
- `--batch_size`: Training batch size (default: 32)
- `--embedding_size`: Size of word embeddings (default: 128)
- `--n_heads`: Number of attention heads (default: 4)
- `--hidden_size`: Size of feed-forward network (default: 256)
- `--n_layers`: Number of encoder/decoder layers (default: 2)
- `--n_epochs`: Number of training epochs (default: 10)
- `--initial_learning_rate`: Starting learning rate (default: 0.001)

Additional Options:
- `--load_checkpoint`: Resume training from a saved checkpoint
- `--attentions_through_time`: Track attention weights for a specific sentence during training
- `--print_every`: Epochs between printing loss info
- `--checkpoint_every`: Epochs between saving checkpoints
- `--bleu_every`: Epochs between computing dev set BLEU scores

### Evaluating Translations

Use `grade.py` to compute BLEU scores for generated translations:

```bash
python grade.py \
    --translations path/to/translations.txt \
    --references path/to/reference_file
```

The reference file should contain source|||target pairs, while the translations file should contain one translation per line.

## Data Format

### Training/Development/Test Files
- Files should contain one sentence pair per line
- Source and target sentences should be separated by `|||`
- BPE tokenization is supported (tokens joined with `@@`)
- Example: `le chat@@s ||| the cat@@s`
