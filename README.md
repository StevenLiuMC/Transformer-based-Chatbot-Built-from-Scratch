# Transformer-based Chatbot

---

# 2025_06_02 Update (Tokenizer Module)
This repository contains a basic implementation of a tokenizer module designed for a Transformer-based chatbot. The code is written in Python using PyTorch.

## 📌 Update Summary

This module includes the definition and implementation of a **SimpleTokenizer** class, which is a lightweight tokenizer capable of processing input text at either the word or character level. It builds a vocabulary from input data, encodes text into token indices for model input, and decodes token indices back into human-readable text.

## Features

- **Device Configuration**: Automatically selects between CPU and GPU based on availability.
- **SimpleTokenizer Class**:
  - **Granularity**: Supports both word-level and character-level tokenization.
  - **Special Tokens**:
    - `<PAD>`: Padding
    - `<UNK>`: Unknown words
    - `<SOS>`: Start of sentence
    - `<EOS>`: End of sentence
  - **Vocabulary Building**: Builds a vocabulary from a list of input text samples, keeping the most frequent tokens (up to a maximum size of 10,000 by default).
  - **Encoding**: Converts raw text into a list of token indices with optional padding and truncation.
  - **Decoding**: Converts a list or tensor of token indices back into a human-readable text string.

## Functions

- `build_vocab(texts, max_vocab_size=10000)`: Builds a vocabulary from the provided texts.
- `encode(text, max_length=None)`: Encodes the given text into a sequence of token indices.
- `decode(indices)`: Decodes a list or tensor of token indices back into readable text.

## Usage

This tokenizer is designed to be used as a preprocessing step for training or inference with Transformer-based models in chatbot or NLP applications.

## Note

- The tokenizer uses simple regular expressions for word splitting and is suitable for basic NLP tasks or prototyping.
- More advanced tokenizers (e.g., subword or BPE) are recommended for production-level applications.
- This project is still developing and this is not a complete file.

---

# 2025_06_10 Update (Positional Encoding, Multi-head Attention, Feed Forward and Encoder Layer)

This update implements several **core components of the Transformer architecture**, focused on the **encoder-side**. The code is written in **PyTorch** with **detailed comments** to support understanding and future development.

---

## 📌 Update Summary

-  **Date:** 2025-06-10  
-  **Modules Implemented:**
  - `PositionalEncoding`
  - `MultiHeadAttention`
  - `FeedForward`
  - `EncoderLayer`

---

## 🧩 Modules Overview

###  1. `PositionalEncoding`

> Adds **sinusoidal positional information** to input embeddings to help the model distinguish the order of tokens.

```python
class PositionalEncoding(nn.Module)
```

**Arguments:**
- `d_model`: Embedding dimension  
- `max_seq_length`: Max sequence length (default: 512)  
- `dropout_rate`: Dropout rate after adding positional encodings  

**Notes:**
- Uses sine for even indices and cosine for odd indices  
- `pe` registered as a non-trainable buffer (`register_buffer`)

---

###  2. `MultiHeadAttention`

> Implements the **scaled dot-product multi-head attention** mechanism.

```python
class MultiHeadAttention(nn.Module)
```

**Arguments:**
- `d_model`: Model dimension  
- `num_heads`: Number of attention heads  
- `dropout_rate`: Dropout rate after softmax  

**Features:**
- Linear projections for `Q`, `K`, `V`  
- Supports attention masks (e.g., for padding or look-ahead)  
- Output projection with residual structure  
- Returns attention weights for visualization

---

###  3. `FeedForward`

> Implements the **position-wise feed-forward network (FFN)** used in Transformer encoders.

```python
class FeedForward(nn.Module)
```

**Arguments:**
- `d_model`: Model dimension  
- `d_ff`: Hidden layer size in FFN  
- `dropout_rate`: Dropout between layers  

**Structure:**
```text
FFN(x) = Linear → ReLU → Dropout → Linear
```

---

###  4. `EncoderLayer`

> Combines attention and feed-forward into a full **Transformer encoder block**.

```python
class EncoderLayer(nn.Module)
```

**Arguments:**
- `d_model`, `num_heads`, `d_ff`, `dropout_rate`

**Structure:**
- Multi-head self-attention → residual → LayerNorm  
- Feed-forward network → residual → LayerNorm

---

##  Design Notes

- **Post-norm style**: LayerNorm applied after residual connection  
- Attention scaling with `sqrt(d_k)` for stability  
- Efficient masking using `masked_fill_` (in-place op)  
- Head splitting via `view()` + `transpose()`  
- Positional encoding stored with `register_buffer()` for correct device movement and saving

---

# 2025_06_19 Update (Decoder Layer, Transformer Assembly, Chatbot Generation)

This update completes the full **Transformer architecture**, adding the decoder stack and a chatbot-ready model. Implemented in **PyTorch** with modular design and detailed comments.

---

## 📌 Update Summary

- **Date:** 2025-06-20  
- **Modules Implemented:**
  - `DecoderLayer`
  - `TransformerChatbot`
  - `generate()` (text generation method within TransformerChatbot)

---

## 🧩 Modules Overview

### 1. `DecoderLayer`

> Implements one decoder block with:
> - Masked self-attention  
> - Cross-attention to encoder output  
> - Feed-forward network  
> All sub-layers use residual connections + LayerNorm.

```python
class DecoderLayer(nn.Module)
```

**Arguments:**
- `d_model`: Model dimension  
- `num_heads`: Number of attention heads  
- `d_ff`: Hidden size in FFN  
- `dropout_rate`: Dropout rate

**Structure:**
- Masked Self-Attn → Add & Norm  
- Cross-Attn → Add & Norm  
- FeedForward → Add & Norm

---

### 2. `TransformerChatbot`

> Full encoder-decoder architecture for chatbot tasks. Includes:
> - Embedding + PositionalEncoding  
> - Stack of encoder/decoder layers  
> - Output projection to vocabulary  
> - Masking for padding and look-ahead
> - Text generation

```python
class TransformerChatbot(nn.Module)
```

**Key Methods:**
- `forward(src, tgt)`: Training-time forward pass  
- `create_padding_mask()`, `create_look_ahead_mask()`: Masking utilities  
- `generate(src, tokenizer)`: Inference-time text generation

---

### 3. `generate()`

> Performs auto-regressive decoding using the trained model.

**Steps:**
1. Encode input with encoder stack  
2. Initialize target with `<SOS>` token  
3. Iteratively decode and sample next token  
4. Stop at `<EOS>` or max length  
5. Return decoded string

**Sampling:**
- Softmax + temperature  
- `torch.multinomial()` for probabilistic token selection

---

## ⚙️ Design Notes

- Post-Norm: `LayerNorm(x + sublayer(x))`  
- Embedding scaled by `√d_model`  
- Xavier (Glorot) initialization for weights  
- `nn.ModuleList` used for stacking layers  
- Comments included for every key step

---

## ✅ Summary of Capabilities

-  Full Transformer encoder-decoder model  
-  Attention masking (padding + autoregressive)  
-  Text generation with sampling  
-  Training and inference support  
-  Clean, modular, research-friendly design


## 📚 References

- Vaswani et al. (2017). *Attention is All You Need*  
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
