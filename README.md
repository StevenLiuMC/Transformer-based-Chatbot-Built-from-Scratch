# Transformer-based Chatbot

# 2025_06_01 Update (Tokenizer Module)
This repository contains a basic implementation of a tokenizer module designed for a Transformer-based chatbot. The code is written in Python using PyTorch.

## Overview

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

## Notes

- The tokenizer uses simple regular expressions for word splitting and is suitable for basic NLP tasks or prototyping.
- More advanced tokenizers (e.g., subword or BPE) are recommended for production-level applications.
- This project is still developing and this is not a complete file.

