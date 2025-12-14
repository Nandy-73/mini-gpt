# Mini GPT – Character-Level Transformer Language Model

This project implements a **decoder-only Transformer (GPT-style) language model** trained from scratch at the **character level** on the Tiny Shakespeare dataset.  
The model learns to predict the next character given a sequence of previous characters and can generate Shakespeare-like text from a short prompt.

---

## Project Objective

The objectives of this mini project are:

- Implement a Transformer-based language model (decoder-only)
- Understand causal self-attention and autoregressive generation
- Train a character-level language model using PyTorch
- Generate coherent Shakespeare-style text

---

## Dataset

- **Name:** Tiny Shakespeare  
- **Size:** ~1 MB  
- **Source:**  
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt  

### Tokenization

- Character-level tokenization
- Each unique character is treated as a token (letters, punctuation, spaces, newline)
- Vocabulary size ≈ **65 characters**
- Input sequences are created using a sliding window of fixed length

---

## Model Architecture

The model follows a **decoder-only Transformer architecture** similar to GPT.

### Components

- Token embeddings
- Positional embeddings
- Stack of Transformer blocks
  - Layer normalization
  - Causal multi-head self-attention
  - Feed-forward neural network (MLP)
  - Residual connections
- Final layer normalization
- Linear language modeling head

### Configuration Used

- Embedding dimension (`d_model`): **384**
- Number of layers: **6**
- Number of attention heads: **6**
- Context length (block size): **128**
- Dropout: **0.1**

---

## Training Setup

### Objective
- Predict the next character in a sequence (autoregressive modeling)

### Loss Function
- Cross-entropy loss

### Optimizer
- Adam optimizer

### Training Parameters

- Batch size: **64**
- Learning rate: **3e-4**
- Training steps: **8000**
- Evaluation interval: **500 steps**
- Device: **GPU (CUDA)**

During training, the validation loss reaches its minimum around step ~2500.  
After this point, the model starts to overfit, so the best checkpoint is saved automatically.

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install torch tqdm


## How to Run the Project

### 1. Install Dependencies  

```bash
pip install torch tqdm

###  2. Download the Dataset
mkdir data
curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt

### 3.python train.py

python train.py

### 4. Generate Text

python generate.py



