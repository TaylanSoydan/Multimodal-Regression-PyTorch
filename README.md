# Multimodal Regression with PyTorch

A from-scratch multimodal neural network that fuses **tabular features**, **text descriptions**, and **spacecraft images** to predict a continuous target. Achieves **R² = 1.0** on the test set — the intended result when all three modalities are correctly integrated.

## The Challenge

Given a dataset of spacecraft candidates with ~40 tabular features, free-text descriptions, and corresponding images, predict a regression target. A tabular-only baseline (CatBoost) plateaus at **R² ≈ 0.86** — the remaining signal lives in the text and images. The challenge tests whether you can design a multimodal architecture that captures complementary information across all three data types.

## Architecture

```
 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │   Text Input   │   │  Image Input   │   │ Tabular Input  │
 │  (5 tokens)    │   │  (64×64 RGB)   │   │  (N features)  │
 └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
         │                    │                     │
   ┌─────▼─────┐       ┌─────▼─────┐        ┌──────▼─────┐
   │ Embedding  │       │  Conv2d   │        │   Linear   │
   │  dim=1     │       │  3→8 ch   │        │  N→20      │
   └─────┬─────┘       │ BatchNorm │        │ LeakyReLU  │
         │             │ LeakyReLU │        └──────┬─────┘
   ┌─────▼─────┐       │ MaxPool2d │               │
   │    GRU    │       ├───────────┤               │
   │  hidden=3 │       │  Conv2d   │               │
   │ LayerNorm │       │  8→16 ch  │               │
   └─────┬─────┘       │ BatchNorm │               │
         │             │ LeakyReLU │               │
         │             │ MaxPool2d │               │
         │             ├───────────┤               │
         │             │ Linear→20 │               │
         │             └─────┬─────┘               │
         │                   │                     │
         └───────┬───────────┴──────────┬──────────┘
                 │      Concatenate     │
                 │     [3 + 20 + 20]    │
                 └──────────┬───────────┘
                      ┌─────▼─────┐
                      │ Linear→4  │
                      │ LeakyReLU │
                      │ Linear→1  │
                      └─────┬─────┘
                            │
                      Prediction (R²=1.0)
```

## Design Decisions

| Choice | Why |
|--------|-----|
| **GRU over BERT** | Descriptions are ~5 tokens — a recurrent layer captures sequence info without the overhead of a pretrained transformer. Also showcases custom training rather than fine-tuning. |
| **Vanilla CNN over ResNet** | Demonstrates understanding of convolutional architectures, custom weight initialization (Kaiming for conv/linear, orthogonal for GRU), and per-branch learning rate tuning. |
| **Separate learning rates** | CNN (0.1), GRU (1e-5), MLP (1e-3) — modalities converge at different rates. Without per-branch LRs, the CNN dominates and the text branch undertrain. |
| **CatBoost baseline** | Establishes that tabular features alone reach R² ≈ 0.86, clearly motivating the multimodal approach. |
| **1 epoch training** | The model converges in a single epoch, demonstrating that the architecture and initialization are well-tuned rather than relying on many passes over the data. |

## What I'd Do Differently Today

- Use a **learning rate scheduler** (e.g. OneCycleLR) instead of fixed per-branch LRs
- Replace the word-level tokenizer with **character n-grams or a small SentencePiece model** for better OOV handling
- Add a **validation split with early stopping** rather than training for a fixed number of epochs
- Experiment with **cross-attention** between modalities instead of late concatenation

## Project Structure

```
├── tabular.py       # CatBoost tabular-only baseline (R² ≈ 0.86)
├── multimodal.py    # Multimodal CNN + GRU + MLP model (R² = 1.0)
├── utils.py         # Data preprocessing utilities
├── run.py           # Runs both models sequentially
├── Dockerfile       # Containerized execution
├── requirements.txt # Python dependencies
├── instructions.txt # Original challenge instructions
├── insights.ipynb   # EDA and analysis notebook
└── step1.ipynb      # Tabular model exploration notebook
```

## Running

**With Docker:**
```bash
docker build -t multimodal-regression .
docker run -it --rm multimodal-regression
```

**Locally:**
```bash
pip install -r requirements.txt
python run.py
```

Note: Both methods require `candidates_data.csv` and the `spacecraft_images/` directory in the project root.
