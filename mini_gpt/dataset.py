import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    A dataset that produces (x, y) pairs of length block_size.
    x = input sequence
    y = same sequence shifted by one character
    """

    def __init__(self, data_ids: torch.Tensor, block_size: int):
        self.data = data_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_vocab_and_encode(text: str):
    """
    Builds a single shared vocabulary for the entire dataset.
    Encodes ALL text into a Tensor of token IDs.
    """
    chars = sorted(list(set(text)))       # list of unique characters
    stoi = {c: i for i, c in enumerate(chars)}   # char -> index
    itos = {i: c for c, i in stoi.items()}       # index -> char
    vocab_size = len(chars)

    # Convert the entire corpus to token IDs
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, stoi, itos, vocab_size


def load_splits(path: str, block_size: int = 128, split: float = 0.9):
    """
    Loads the dataset, builds ONE vocab, encodes everything,
    then splits into train/val sets.
    """

    # Read text
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build shared vocabulary for train & val
    ids, stoi, itos, vocab_size = build_vocab_and_encode(text)

    # Train/val split
    n = int(split * len(ids))
    train_ids = ids[:n]
    val_ids   = ids[n:]

    # Create dataset objects
    train_ds = CharDataset(train_ids, block_size)
    val_ds   = CharDataset(val_ids, block_size)

    # Attach vocab information to BOTH datasets
    train_ds.stoi, train_ds.itos, train_ds.vocab_size = stoi, itos, vocab_size
    val_ds.stoi,   val_ds.itos,   val_ds.vocab_size   = stoi, itos, vocab_size

    return train_ds, val_ds
