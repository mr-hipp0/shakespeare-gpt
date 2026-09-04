# shakespeare-gpt

A PyTorch implementation of a Decoder-only Transformer (GPT) language model trained on the **Tiny Shakespeare** dataset from scratch.

---

## Features

* **Character-Level Tokenization**: Map individual characters directly to integers for simple, lightweight pre-processing without external tokenizers.
* **Decoder-Only Transformer**: Complete implementation of masked multi-head self-attention, positional embeddings, layer normalization, residual connections, and feed-forward networks.
* **Incremental Progression**: Clean code structure moving from a baseline Bigram model up to the full GPT architecture.
* **Autoregressive Text Generation**: Includes sampling functions to prompt the model and produce streamable text outputs.

---

## Model Architecture

The transformer implementation follows the standard decoder-only architecture:

1. **Token & Positional Embeddings**: Input character IDs and position indices are embedded into vectors.
2. **Masked Multi-Head Self-Attention**: Queries, Keys, and Values compute scaled dot-product attention with a triangular causal mask to prevent attending to future tokens.
3. **Feed-Forward Layers**: Multi-Layer Perceptron (MLP) expanding features by a factor of 4x before projecting back.
4. **Pre-Layer Normalization & Residuals**: Residual connections surround each Attention and Feed-Forward block with Pre-LayerNorm applied prior to transformations.
5. **Linear Output Head**: Maps hidden representations to character vocabulary logits.

---

## Hyperparameters

Default configurations set in model training:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `batch_size` | `64` | Independent sequences processed in parallel |
| `block_size` | `256` | Maximum context length for predictions |
| `max_iters` | `5000` | Total training iterations |
| `eval_interval` | `500` | Frequency of validation loss evaluation |
| `learning_rate` | `3e-4` | Optimizer learning rate (AdamW) |
| `n_embd` | `384` | Embedding dimensionality |
| `n_head` | `6` | Number of parallel attention heads |
| `n_layer` | `6` | Number of Transformer block layers |
| `dropout` | `0.2` | Regularization dropout rate |

---

## Sample Generated Output

After training for ~5,000 steps, the model produces structurally valid plays and dialogues:

> **KING RICHARD III:**
> What, shall we go to London once again?
> I'll pay the debt that thy affection owes,
> And leave the kingdom to a worthy heir.
> 
> **DUKE OF YORK:**
> Speak on, my lord; the truth shall set thee free!

---

## Resources & Credits

* **YouTube Tutorial**: [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1es) by Andrej Karpathy
* **Reference Code**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
* **Dataset**: [Tiny Shakespeare Dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
