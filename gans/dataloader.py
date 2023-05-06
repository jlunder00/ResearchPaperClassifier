from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
import json
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class AbstractDataset(Dataset):
    def __init__(self, papers_info_fname, ntokens, device):
        self.papers_info_fname = papers_info_fname
        with open(papers_info_fname, 'r') as fin:
            self.papers_info = json.loads(fin.read()) 
        self.tokenizer = get_tokenizer("basic_english")
        self.ntokens = ntokens
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.get_text_only_iter()), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.device = device

    def __len__(self):
        return len(self.papers_info)

    def __getitem__(self, idx):
        item = self.papers_info[idx]
        text_tensor = self.get_text_representation(item['abstract'])
        return text_tensor

    def get_text_only_iter(self):
        for item in self.papers_info:
            yield item['abstract']

    def get_text_representation(self, text):
        vectorized_text = self.vocab(self.tokenizer(text))
        length = len(vectorized_text)
        while length < self.ntokens:
            vectorized_text.append(0)
        if length > self.ntokens:
            vectorized_text = vectorized_text[0:self.ntokens]
        tensorified = torch.tensor(vectorized_text, dtype=torch.long).to(self.device)
        return tensorified

    # def data_process(self, raw_text_iter):
    #     data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    #     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))








