import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import load_splits
from model import MiniGPT
from utils import estimate_loss, set_seed

def main():
    set_seed(SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"

    # Data
    train_ds, val_ds = load_splits(DATA_PATH, BLOCK_SIZE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # Model
    model = MiniGPT(
        vocab_size=train_ds.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    step = 0
    best_val = float("inf")

    pbar = tqdm(total=MAX_STEPS, desc="Training")
    while step < MAX_STEPS:
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)                          # B,T,V
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            pbar.update(1)

            if step % EVAL_INTERVAL == 0:
                val_loss = estimate_loss(model, val_dl, device)
                print(f"\nstep {step} | train {loss.item():.3f} | val {val_loss:.3f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "vocab_size": train_ds.vocab_size,
                        "itos": train_ds.itos,
                        "stoi": train_ds.stoi,
                        "config": {
                            "block_size": BLOCK_SIZE,
                            "d_model": D_MODEL,
                            "n_layers": N_LAYERS,
                            "n_heads": N_HEADS,
                            "dropout": DROPOUT
                        }
                    }, CHECKPOINT)
                    print(f"✅ saved checkpoint to {CHECKPOINT}")
            if step >= MAX_STEPS:
                break
    pbar.close()

if __name__ == "__main__":
    # Make sure data file exists
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download tiny Shakespeare to data/input.txt first."
        )
    main()
