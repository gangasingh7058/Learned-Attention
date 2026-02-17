# Learned Attention 🎭

A decoder-only Transformer with a **learned soft attention mask** — each layer dynamically predicts *what to attend to* based on the input, rather than relying solely on a fixed causal mask.

## Core Idea

Standard Transformers use a static causal mask (lower-triangular) that uniformly allows attending to all past tokens. This model adds a **learnable, data-dependent soft mask** that modulates attention weights *after* softmax. Each layer contains a `PredictMask` sub-network that:

1. Runs self-attention over the input to understand context
2. Projects to a `(B, Seq, Seq)` mask via a linear layer
3. Sharpens the mask with a scaled sigmoid
4. Multiplies it element-wise with the main attention weights (post-softmax)

This allows the model to learn **per-layer, input-dependent attention patterns** — selectively amplifying or suppressing token relationships.

## Architecture

```
Input → Embedding (weight-tied) → [ProcessWithLearnedMask × N] → RMSNorm → Linear → Logits
                                         │
                                         ├── PredictMask: Self-Attn → Linear → Sigmoid → soft_mask
                                         ├── Main Attention (RoPE + causal + soft_mask)
                                         └── SwiGLU FeedForward
```

**Key components:**
- **RoPE** (Rotary Position Embeddings)
- **SwiGLU** FFN (gated variant with SiLU activation)
- **RMSNorm** (pre-norm architecture)
- **Weight tying** between embedding and output projection

## Default Configuration

| Parameter | Value |
|---|---|
| `d_model` | 256 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| `max_Seq_len` | 128 |
| `vocab_size` | 50,257 (GPT-2 BPE) |
| `ffn_dim_multiplier` | 1 |
| Total Parameters | ~15.7M |

## Project Structure

```
├── src/
│   ├── model_args.py     # Model hyperparameters (dataclass)
│   ├── model.py          # Transformer + PredictMask architecture
│   ├── dataset.py        # Shakespeare dataset with tiktoken BPE
│   ├── train.py          # Training loop (AdamW, cosine LR, checkpointing)
│   └── inference.py      # Autoregressive text generation
├── Datasets/
│   └── Shakespear_dataset/
└── checkpoints/          # Saved model checkpoints & loss histograms
```

## Quick Start

### Requirements

```bash
pip install torch tiktoken matplotlib tqdm
```

### Train

```bash
cd src
python train.py
```

Training saves checkpoints and loss distribution histograms to `checkpoints/` after every epoch.

### Generate Text

```bash
cd src
python inference.py
```

Generates text from the prompt `"To be or not to be, that is"` using the trained model.

## How the Learned Mask Works

In each Transformer layer (`ProcessWithLearnedMask`):

```
                     ┌──────────────────────┐
                     │     PredictMask       │
    x ──────────────►│  Self-Attn + Linear   │──► soft_mask (B, 1, Seq, Seq)
                     │  + Scaled Sigmoid     │
                     └──────────────────────┘
                                │
    x ──► RMSNorm ──► Multi-Head Attention ◄─── hard causal mask
                          │         ▲                    
                          │    soft_mask (post-softmax modulation)
                          ▼
                     + residual
                          │
                     ──► RMSNorm ──► SwiGLU FFN ──► + residual ──► output
```

The soft mask is applied **after softmax** and the attention is **re-normalized**, keeping the distribution valid while allowing differentiable modulation.

## License

MIT
