import torch
import torch.nn.functional as F
from config import *
from model import MiniGPT

def sample_logits(logits, temperature=0.6, top_k=None):
    logits = logits / max(temperature, 1e-8)
    if top_k is not None and top_k > 0:
        v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, ix, v)
        logits = mask
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

@torch.no_grad()
def generate(model, idx, max_new_tokens=300, block_size=128, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]  # last step
        next_id = sample_logits(logits, temperature, top_k)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg  = ckpt["config"]
    model = MiniGPT(
        vocab_size=ckpt["vocab_size"],
        block_size=cfg["block_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stoi"], ckpt["itos"], cfg["block_size"]

def encode(s, stoi): return torch.tensor([[stoi[c] for c in s]], dtype=torch.long)
def decode(t, itos): return "".join(itos[i] for i in t.tolist())

if __name__ == "__main__":
    device = DEVICE if torch.cuda.is_available() else "cpu"
    model, stoi, itos, block_size = load_checkpoint(CHECKPOINT)
    model.to(device)

    prompt = "O God, O God!"
    idx = encode(prompt, stoi).to(device)

    out = generate(
        model, idx, max_new_tokens=400,
        block_size=block_size, temperature=TEMPERATURE, top_k=TOP_K
    )
    gen = decode(out[0, len(prompt):].cpu(), itos)
    print(prompt + gen)
