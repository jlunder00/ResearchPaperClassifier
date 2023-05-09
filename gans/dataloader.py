from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
import json
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class AbstractDataset(Dataset):
    def __init__(self, papers_info_fname, title_tokens, abstract_tokens, device):
        self.papers_info_fname = papers_info_fname
        with open(papers_info_fname, 'r') as fin:
            papers_info = json.load(fin) 
        if isinstance(papers_info[0], list):
            self.papers_info = [item for sublist in papers_info for item in sublist]
        else:
            self.papers_info = papers_info
        self.tokenizer = get_tokenizer("basic_english")
        self.abstract_tokens = abstract_tokens
        self.title_tokens = title_tokens
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.get_text_only_iter()), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.device = device

    def __len__(self):
        return len(self.papers_info)

    def __getitem__(self, idx):
        item = self.papers_info[idx]
        text_tensor = self.get_text_representation(item['abstract'], self.abstract_tokens)
        title_tensor = self.get_text_representation(item['title'], self.title_tokens)
        return title_tensor, text_tensor
    
    def get_text_only_iter(self):
        for item in self.papers_info:
            yield item['title']+'. '+item['abstract']

    def get_text_representation(self, text, ntokens):
        vectorized_text = self.vocab(self.tokenizer(text))
        length = len(vectorized_text)
        while length < ntokens:
            vectorized_text.append(0)
            length = len(vectorized_text)
        if length > ntokens:
            vectorized_text = vectorized_text[0:ntokens]
        tensorified = torch.tensor(vectorized_text, dtype=torch.float32).to(self.device)
        return tensorified

    # def data_process(self, raw_text_iter):
    #     data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    #     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def collate_fn(batch):
    titles = [item[0] for item in batch]
    abstracts = [item[1] for item in batch]

    # Convert inputs and labels into tensors
    titles = torch.stack(titles) 
    abstracts = torch.stack(abstracts)

    return {'titles': titles, 'abstracts': abstracts}

def collate_fn_GPT2(batch):
    titles = [item['titles'] for item in batch]
    abstracts = [item['abstracts']for item in batch]
    titles_ids = [torch.tensor(item['titles_ids'], dtype=torch.long)  for item in batch]
    abstracts_ids = [torch.tensor(item['abstracts_ids'], dtype=torch.long)  for item in batch]
    titles_attention_masks = [torch.tensor(item['titles_attention_masks'], dtype=torch.long)  for item in batch]
    abstracts_attention_masks = [torch.tensor(item['abstracts_attention_masks'], dtype=torch.long)  for item in batch]
    
    
    
    titles_ids_squeezed = torch.stack(titles_ids).squeeze(1)
    abstracts_ids_squeezed = torch.stack(abstracts_ids).squeeze(1)
    titles_attention_masks_squeezed = torch.stack(titles_attention_masks).squeeze(1)
    abstracts_attention_masks_squeezed = torch.stack(abstracts_attention_masks).squeeze(1)
    return {
        'titles':titles, 
        'abstracts':abstracts, 
        'titles_ids':titles_ids_squeezed, 
        'abstracts_ids':abstracts_ids_squeezed,
        'titles_attention_masks':titles_attention_masks_squeezed, 
        'abstracts_attention_masks':abstracts_attention_masks_squeezed
        }


if __name__ == "__main__":
    print("shut up")







