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

---

# 2025_06_25 Update (Full Chatbot Pipeline with Real Dataset & Training Utilities)

This update implements a complete end-to-end **Transformer-based chatbot pipeline**, from dataset preparation to model training and inference. Designed in **PyTorch** with clear modular structure and real dialogue data from Hugging Face.

---

## 📌 Update Summary

- **Date:** 2025-06-25  
- **New Components:**
  - `ChatDataset`: Custom Dataset class  
  - `train_epoch()` & `evaluate()`: Training loop helpers  
  - `load_real_data()`: Preprocessing HuggingFace persona-chat  
  - `main()`: Full training script with validation & saving  
  - `load_model()` & `chat_with_bot()`: Inference functions
- **Components Changed:**
  - `encode` method under `SimpleTokenizer`

---

## 🧩 Modules Overview

### 1. Updated `encode` method under `SimpleTokenizer`

> Here’s how the previous encode method is implemented:

```python
    def encode(self, text, max_length = None):
      
      tokens = self._tokenize(text)
      indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

      if max_length:
        # Truncate and remain one spot for EOS
        if len(indices) >= max_length - 1:
          indices = indices[:max_length-1]

        # If sequence is not long enough, add EOS first to tell the model that the useful information has ended and then add PAD to make the indices dimension unified.
        indices.append(self.word2idx['<EOS>'])

        if len(indices) < max_length:
          indices += [self.word2idx['<PAD>']] * (max_length - len(indices))
        else:
        # If there's no limitation on length, simply add EOS
          indices.append(self.word2idx['<EOS>'])

      return indices
```
 - However, when I actually used the encode method, I found that as soon as the input text is shorter than max_length and requires <PAD> tokens, the position of <EOS> in the word-ID sequence becomes unpredictable. This creates a lot of headaches in ChatDataset—for example, tgt_input must not include <EOS> while tgt_output must. If the <EOS> token’s location is uncertain, later edits become unnecessarily complicated. To solve this, I modified the encode method in SimpleTokenizer so that it can be instructed whether to include <SOS> or <EOS> when encoding, and I provided sensible default arguments so that calls to encode without explicit <SOS>/<EOS> flags remain fully backward-compatible.

> The latest version of the encode method is implemented as follows:

```python
    def encode(
        self,
        text: str,
        max_length: int = None,
        add_sos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
       
        tokens = self._tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

        # Spots reserved for <SOS> and <EOS>
        reserve = int(add_sos) + int(add_eos)

        # Make sure token ≤ max_length - reserve
        if max_length and len(indices) > max_length - reserve:
            indices = indices[: max_length - reserve]

        # Insert SOS
        if add_sos:
            indices = [self.word2idx["<SOS>"]] + indices  # Add SOS

        # Insert EOS
        if add_eos:
            indices.append(self.word2idx["<EOS>"])  # Add EOS

        # PAD 到固定长度 Pad to max_length
        if max_length and len(indices) < max_length:
            pad_id = self.word2idx["<PAD>"]
            indices += [pad_id] * (max_length - len(indices))

        return indices  # Return the index list contains word ID
```

---

### 2. `ChatDataset`

> Purpose of This Class
> - Convert raw (context, response) text pairs into three ID tensors for the model:
>   - src: input sequence (no <SOS>, with <EOS>) 
>   - tgt_input: decoder input (with <SOS>, no <EOS>)
>   - tgt_output: decoder target (no <SOS>, with <EOS>)
> How It’s Implemented?
> - __init__ stores the data list, tokenizer, and max_length
> - __len__ returns the total number of examples
> - __getitem__ retrieves the text pair at index idx, calls tokenizer.encode(...) three times (with appropriate add_sos/add_eos flags), and wraps each ID list in a torch.long tensor

---

### 3. `train_epoch() & evaluate()`

> Purpose of This Part:
> - train_epoch(): Run one full training pass and return the average loss.
> - evaluate(): Compute validation loss without updating model weights.
> How They’re Implemented?
> - Both accept (model, dataloader, criterion, device)
> - train_epoch():
>   - Set model.train()
>   - Loop over batches: zero grads → forward → reshape outputs & targets → compute loss → backpropagate → clip gradients → optimizer step → accumulate loss
> - evaluate():
>   -  Set model.eval() + torch.no_grad()
>   -  Loop over batches: forward → reshape → compute loss → accumulate
>   -  Return average loss

---

### 4. `load_real_data()`

> Purpose of This Part:
> - Download and preprocess the “persona-chat” dataset into flat (context, reply) lists, then split into train/validation sets.
> How They’re Implemented?
> - Use load_dataset("Cynaptics/persona-chat", split="train").
> - Merge all persona_b lines into a single character description.
> - Tag each dialogue turn as <USER> (Persona A) or <BOT> (Persona B) and join into conversation history.
> - Combine persona + history → context; take reference → reply.
> - Shuffle, optionally trim to max_pairs, and split by split_ratio.

---

### 5. `main()`

> What Happened within This Part:
> - Orchestrate the entire training pipeline: data loading, tokenizer & loader setup, model/optimizer/loss configuration, epoch loops, and checkpoint saving.
> How They’re Implemented?
> - Define hyperparameters (batch size, LR, epochs, model dims).
> - Call load_real_data() for train/val pairs.
> - Build vocabulary with SimpleTokenizer.
> - Wrap each split in ChatDataset + DataLoader.
> - Instantiate TransformerChatbot, CrossEntropyLoss (with label smoothing and pad masking), Adam optimizer, and LR scheduler.
> - For each epoch: run train_epoch(), then evaluate(), step the scheduler, and save the model if validation loss improves.

---

### 6. `load_model() & chat_with_bot()`

> Functions Completed:
> - load_model(): Restore model weights, tokenizer, and hyperparameters from a saved checkpoint.
> - chat_with_bot(): Launch an interactive command-line loop for user–model conversation.
> How They’re Implemented?
> - load_model():
>   - Load .pth file with torch.load()
>   - Extract hyperparameters, build a new TransformerChatbot instance, and load state_dict
>   - Return the model (in eval() mode) and the tokenizer
> - chat_with_bot():
>   - Prompt the user in a loop until they type “quit”.
>   - Encode user text (<EOS> appended), call model.generate() for autoregressive decoding, and print the decoded response.

---

## 📚 References

- Vaswani et al. (2017). *Attention is All You Need*  
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
