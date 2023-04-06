import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset 
import math

import pandas
import matplotlib.pyplot as plt


class SimpleGenerator(nn.Module):
    def __init__(self, ntokens, device):

        super().__init__()
        self.ntokens = ntokens
        self.model = nn.Sequential(
            nn.Linear(1, int(ntokens/4)),
            nn.ReLU(),
            nn.Linear(int(ntokens/4), int(ntokens/2)),
            nn.ReLU(),
            nn.Linear(int(ntokens/2), int(ntokens/2)),
            nn.ReLU(),
            nn.Linear(int(ntokens/2), int(ntokens)),
            nn.ReLU()
        )
        self.device = device
        self.model.to(device)
        self.loss_function = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)
    
    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        
        d_output = D.forward(g_output)
        
        loss = D.loss_function(d_output, targets)
        
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))

# class TransformerGenerator(nn.Module):
#     def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5, num_discriminator_features: int = 1):
#         super().__init__()
#         self.model_type = 'Transformer'
#         self.PositionalEncoding(d_model, dropout)
#         encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Embedding(num_discriminator_features, d_model)
#         self.d_model = d_model
#         self.decoder = nn.Linear(d_model, ntoken)
#         self.num_discriminator_features = num_discriminator_features

#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
#         src = self.encoder(src) * math.sqrt(self.d_model)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.decoder(output)
#         return output

#     def generate_square_subsequent_mask(sz: int) -> Tensor:
#         """Generates an upper-triangular matrix of -inf, with zeros on diag."""
#         return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


