# Training & model config you can tweak
BLOCK_SIZE   = 128
BATCH_SIZE   = 64
D_MODEL      = 384
N_HEADS      = 6
N_LAYERS     = 6
DROPOUT      = 0.1
LR           = 3e-4
WEIGHT_DECAY = 0.0
MAX_STEPS    = 8000        
EVAL_INTERVAL= 500
DEVICE       = "cuda"
SEED         = 42
TOP_K        = 50
TEMPERATURE  = 0.9
CHECKPOINT   = "minigpt.pt"
DATA_PATH    = "data/input.txt"
