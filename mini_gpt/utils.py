import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def estimate_loss(model, data_loader, device="cuda"):
    model.eval()
    total = 0.0
    n = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)

def perplexity_from_loss(loss):
    return math.exp(loss)

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
