# NovaLM

A small GPT-style language model built from scratch in PyTorch and trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
Every component (embeddings, positional encoding, self-attention, multi-head
attention, feed-forward, transformer blocks) lives in its own module under `model/`.

## Model

Defaults from `configs/config.py`:

| Setting | Value |
|---|---|
| Vocabulary | 16,000 tokens (`tokenizer/tokenizer.json`) |
| Context length | 256 |
| Embedding size | 256 |
| Layers / heads | 6 / 8 |
| Feed-forward size | 1024 |
| Dropout | 0.1 |

## Project layout

```
configs/      model, training, data and generation settings
model/        transformer building blocks and the NovaLM model
dataset/      PyTorch dataset over tokenized data
training/     training loop
inference/    text generation
scripts/      data prep, quality checks, the web GUI and component tests
evaluation/   training history (CSV) and loss plots
tokenizer/    trained tokenizer
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch tokenizers matplotlib
```

The dataset and checkpoints are not in the repo (they're hundreds of MB).
Download `TinyStories-train.txt` from Hugging Face into `data/raw/`, then:

```bash
python prepare_data.py                              # clean, tokenize and split
python train.py                                     # train (or: --steps 500)
python generate.py --prompt "The scientist discovered"
python gui.py --port 8080                           # NovaLM Studio in the browser
```

Checkpoints are written to `checkpoints/`.

## Tests

The `scripts/test_*.py` files check single parts (tokenizer, embeddings, loss,
one training step):

```bash
python -m scripts.test_tokenizer
```
