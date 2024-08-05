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

class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.sqrt_d_model = sqrt(d_model)
        self.embed = nn.Embedding(vocab, d_model)
    def forward(self, src):
        return self.embed(src) * sqrt_d_model
        # this results in embedding vectors whose standard deviation is equal to sqrt_d_model

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, p_dropout, max_len=5000):
        super(self).__init__()
        self.dropout = nn.Dropout(p_dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len).unsqueeze(1)
        # The divisor for both functions is the same
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # we don't need the params to be learnable so we set requires_grad to false
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
        
class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clone(layer, n_layers)
    def forward(self, src):
        encoded = src
        for layer in layers:
            encoded = layer(src)
        return src

# The encoder layer consists of two parts the Multi Head Attention (MHA) and a Feed-Forward Network (FFN)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, p_dropout, mha, ffn):
        super().__init__();
        self.MHA = mha 
        self.FFN = ffn 

# FFN(x)=max(0,x . W1 + b1) . W2 + b2
# We use dropout to avoid overfitting

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, p_dropout):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ffn)
        self.W2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(p_dropout)
    def forward(self, src):
        return self.W2(self.dropout(self.W1(src)))

class MHA(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class Generator(nn.Module):
    def __init__(self, d_model: int, trg_vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1);
    def forward(self, src):
        return self.log_softmax(self.linear(src))











def clone(module, n):
    # This is used to create copies of the encoder and decoder layers
    return nn.ModuleList([deepcopy(module) for _ in range(n)])
