import torch
import wandb
from discriminator import SimpleDiscriminator, GPT2Discriminator
from generator import SimpleGenerator, GPT2Generator
from dataloader import AbstractDataset, collate_fn, collate_fn_GPT2
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import get_linear_schedule_with_warmup, AdamW, AutoTokenizer




def train_one_epoch(loader, D, G):
    for i, data in enumerate(loader):
        titles = data['titles']
        abstracts = data['abstracts']
        D.train(abstracts, torch.FloatTensor([1.0, 0.0]).to('cuda'))
    
        D.train(G.forward(titles.detach()), torch.FloatTensor([0.0, 1.0]).to('cuda'))
    
        G.train(D, titles, torch.FloatTensor([1.0, 0.0]).to('cuda'))

    D.plot_progress()
    
    
def train_one_epoch_gpt2(loader, D, G, optimizer_, scheduler_, device_):
    D.model.to(device_)
    G.model.to(device_)
    for i, data in enumerate(loader):
        data = {k:v.to(device_) if k not in ['titles', 'abstracts'] else v for k,v in data.items()}
        titles = data['titles']
        abstracts = data['abstracts']
        titles_ids = data['titles_ids']
        abstracts_ids = data['abstracts_ids']
        titles_attention_masks = data['titles_attention_masks']
        abstracts_attention_masks = data['abstracts_attention_masks']
        real_input = {'input_ids':abstracts_ids, 'attention_mask':abstracts_attention_masks}
        generators_input = {'input_ids':titles_ids, 'attention_mask':titles_attention_masks}
        real_labels = {'labels':torch.tensor([1 for i in range(len(titles))], dtype=torch.long).to(device_)}
        fake_labels = {'labels':torch.tensor([0 for i in range(len(titles))], dtype=torch.long).to(device_)}
        real_input.update(real_labels)
        
        print(D.model.config)
        D.train(real_input,optimizer_)
        fake_abstracts = G.model.generate(generators_input['input_ids'].detach())
        print('FAKE ABSTRACTS', fake_abstracts)
        D.tokenizer(G.tokenizer.decode(G.model.generate(generators_input['input_ids'])), 
                    padding="max_length", truncation=True, return_tensors='pt', max_length=512)
        
    
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
    
    train_loader = DataLoader(train_dataset['train'], collate_fn=collate_fn_GPT2, batch_size=1)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = "left"
    
    
    D = GPT2Discriminator('distilbert-base-uncased', tokenizer=tokenizer)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    G = GPT2Generator('gpt2-finetuning_again_best/checkpoint-1000', tokenizer=tokenizer)
    
    Doptimizer = AdamW(D.model.parameters(),
                  lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(train_loader) * EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(Doptimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    train_one_epoch_gpt2(train_loader, D, G, Doptimizer, scheduler, 'cuda')
    
    
    
    
    

    
    
    
def train_wandb(config):
    with wandb.init(project='research-paper-classifier', config=config):
        train()
        
def train_wandb_GPT2(config):
    with wandb.init(project='research-paper-classifier', config=config):
        train_gpt2()
        
if __name__ == '__main__':
    
    import torch
    import wandb
    #from training import train_wandb_GPT2, train_gpt2
    config = dict(
        epochs = 50,
        batch_size = 4,
        learning_rate = 0.00001,
        architecture="NN",
        abstract_tokens = 300,
        title_tokens = 16
    )
    train_wandb_GPT2(config)
    