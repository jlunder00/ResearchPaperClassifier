import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset 
import math
from transformers import GPT2ForSequenceClassification, TrainingArguments, Trainer, GPT2Config, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer, DistilBertConfig
import gc

import pandas
import matplotlib.pyplot as plt


class SimpleGenerator(nn.Module):
    def __init__(self, title_tokens, abstract_tokens, device):

        super().__init__()
        self.abstract_tokens = abstract_tokens
        self.title_tokens = title_tokens
        self.model = nn.Sequential(
            nn.Linear(self.title_tokens, int(self.abstract_tokens/4)),
            nn.ReLU(),
            nn.Linear(int(self.abstract_tokens/4), int(self.abstract_tokens/2)),
            nn.ReLU(),
            nn.Linear(int(self.abstract_tokens/2), int(self.abstract_tokens/2)),
            nn.ReLU(),
            nn.Linear(int(self.abstract_tokens/2), int(self.abstract_tokens)),
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
        
from transformers import AutoModelWithLMHead

class GPT2Generator():
    def __init__(self, path, tokenizer):
        self.config = GPT2Config.from_pretrained(path)
        self.config.pad_token_id = tokenizer.pad_token_id
        self.config.pad_token = tokenizer.pad_token
        print(self.config.pad_token)
        self.tokenizer = tokenizer
        self.model = AutoModelWithLMHead.from_pretrained(path, config=self.config)    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.total_loss = 0
        self.running_loss = 0
        
    def forward(self, inputs):
        return self.model(input_ids = inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    def train(self, D, inputs, optimizer_, scheduler_):
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs[0], dim=-1).squeeze()
        #fake_abstracts_ids = [torch.multinomial(item, num_samples=1).squeeze() for item in probs]
        #decoded = [self.tokenizer.decode(item) for item in fake_abstracts_ids]
        #fake_input_list = [D.tokenizer(item, padding="max_length", truncation=True, return_tensors='pt', max_length=512) for item in decoded]
        #fake_input = {}
        #for item in fake_input_list:
        #    for k, v in item.items():
        #        if k not in fake_input.keys():
        #            fake_input[k] = []
        #        fake_input[k].append(v)
        #fake_input = {k:torch.stack(v).squeeze() for k,v in fake_input.items()}
        
        #for item in fake_input_list:
        #    del item
        #for item in fake_abstracts_ids:
        #    del item
        #for item in decoded:
        #    del item
        
        fake_abstracts_ids = torch.stack([torch.multinomial(probs[i], num_samples=1).squeeze() for i in range(len(probs))])
        decoded = [self.tokenizer.decode(ids) for ids in fake_abstracts_ids]
        fake_input_list = [D.tokenizer(item, padding="max_length", truncation=True, return_tensors='pt', max_length=512) for item in decoded]
        fake_input = {}
        for item in fake_input_list:
            for k, v in item.items():
                if k not in fake_input.keys():
                    fake_input[k] = []
                fake_input[k].append(v)
        fake_input = {k:torch.stack(v).squeeze() for k,v in fake_input.items()}
            
        #fake_abstracts = []
        #for i in range(len(probs)):
        #    fake_abstract_ids = torch.multinomial(probs[i], num_samples=1).squeeze()
        #    fake_abstract = self.tokenizer.decode(fake_abstract_ids)
        #    fake_input = D.tokenizer(fake_abstract, padding="max_length", truncation=True, return_tensors='pt', max_length=D.tokenizer.model_max_length)
        #    fake_abstracts.append(fake_input)
        #fake_input = {k:torch.stack([item[k] for item in fake_abstracts]).squeeze() for k in fake_abstracts[0].keys()}
        #fake_input = {k:v.to(self.device) for k,v in fake_input.items()}
        
        discriminator_output = D.model(input_ids=fake_input['input_ids'].detach(), attention_mask=fake_input['attention_mask'])
        
        labels = torch.stack([torch.tensor([0,1],dtype=torch.float).to(self.model.device) for i in range(len(fake_input['input_ids']))])
        loss = torch.nn.functional.mse_loss(discriminator_output.logits.to(self.model.device), labels)
        self.total_loss += loss.item()
        self.running_loss += loss.item()
        return loss
    

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


