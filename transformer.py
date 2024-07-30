import torch
from torch import nn
import math

class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
    def forward(self, )
        
