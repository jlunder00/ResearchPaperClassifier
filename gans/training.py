import torch
import wandb
from discriminator import SimpleDiscriminator
from generator import SimpleGenerator
from dataloader import AbstractDataset, collate_fn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk




def train_one_epoch(loader, D, G):
    for i, data in enumerate(loader):
        titles = data['titles']
        abstracts = data['abstracts']
        D.train(abstracts, torch.FloatTensor([1.0, 0.0]).to('cuda'))
    
        D.train(G.forward(titles.detach()), torch.FloatTensor([0.0, 1.0]).to('cuda'))
    
        G.train(D, titles, torch.FloatTensor([1.0, 0.0]).to('cuda'))

    D.plot_progress()
    
    
def train_one_epoch_gpt2(loader, D, G, optimizer_, scheduler_, device_):
    for i, data in enumerate(loader):
        titles = data['titles']
        abstracts = data['abstracts']
        titles_ids = data['titles_ids']
        abstracts_ids = data['abstracts_ids']
        titles_attention_masks = data['titles_attention_masks']
        abstracts_attention_masks = data['abstracts_attention_masks']
        real_input = {'text':abstracts, 'input_ids':abstracts_ids, 'attention_mask':abstracts_attention_masks}
        generators_input = {'text':titles, 'input_ids':titles_ids, 'attention_mask':titles_attention_masks}
        real_labels = {'labels':torch.tensor([1 for i in range(len(titles))], dtype=torch.long)}
        fake_labels = {'labels':torch.tensor([0 for i in range(len(titles))], dtype=torch.long)}
        real_input.extend(real_labels)
        D.train(real_input,optimizer_)
    
def train():
    torch.set_printoptions(precision=10)
    EPOCHS = wandb.config.epochs
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    abstract_tokens = wandb.config.abstract_tokens
    title_tokens = wandb.config.title_tokens
    data = AbstractDataset('data/small_with_titles_and_abstracts.json', abstract_tokens=abstract_tokens, title_tokens=title_tokens, device='cuda')
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    D = SimpleDiscriminator(abstract_tokens=abstract_tokens, device='cuda')
    G = SimpleGenerator(title_tokens=title_tokens, abstract_tokens=abstract_tokens, device='cuda')
    
def train_gpt2():
    torch.set_printoptions(precision=10)
    EPOCHS = wandb.config.epochs
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    
    train_dataset = load_from_disk('data/train_tokenized_json_small')
    valid_dataset = load_from_disk('data/valid_tokenized_json_small')
    test_dataset = load_from_disk('data/test_tokenized_json_small')
    
    
    
def train_wandb(config):
    with wandb.init(project='research-paper-classifier', config=config):
        train()
    