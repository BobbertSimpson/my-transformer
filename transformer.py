from copy import deepcopy
import torch
from torch import nn
import math

class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, encoder: Encoder, decoder: Decoder, src_vocab_size: int, tgt_vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model # The dimention of the model (the size of the embeddings)
        self.src_embed = Embedding(src_vocab_size, d_model) # Get the embeddings for the source
        self.tgt_embed = Embedding(tgt_vocab_size, d_model) # Get the embeddings for the target
        self.generator = Generator(model_dimension, trg_vocab_size) # Applies nn.Linear to the output of the final decoder layer and returns the probabilities using the sigmoid function
        self.init_params() # Initialize all of the weights of the model

    def forward(self, src, tg):
             
    def init_params(self):
        for param in self.parameters():
            if param.dim() > 1: # nn.init.xavier_transform_ only works for dim > 1
                nn.init.xavier_transform_(param)

class Generator(nn.Module):
    def __init__(self, d_model: int, trg_vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1);
    def forward(self):
        
class Encoder(nn.Module):
    def __init__(self, d_model: int, n: int):
        self.layers = 














def clones(module, n):
    # This is used to create copies of the encoder and decoder layers
    return nn.ModuleList([deepcopy(module) for _ in range(n)])
