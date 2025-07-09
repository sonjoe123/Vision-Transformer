import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Constants
EMBED_DIM = 512
HEADS = 8
HEAD_DIM = EMBED_DIM / HEADS

# Multihead Attention Function:

class muliheadattention(nn.Module):
    def __init__ (self, num_heads, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim