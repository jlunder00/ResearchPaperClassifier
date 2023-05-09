import torch
import wandb
from discriminator import SimpleDiscriminator, GPT2Discriminator
from generator import SimpleGenerator, GPT2Generator
from dataloader import AbstractDataset, collate_fn, collate_fn_GPT2
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import get_linear_schedule_with_warmup, AdamW, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime




def train_one_epoch(loader, D, G):
    for i, data in enumerate(loader):
        titles = data['titles']
        abstracts = data['abstracts']
        D.train(abstracts, torch.FloatTensor([1.0, 0.0]).to('cuda'))
    
        D.train(G.forward(titles.detach()), torch.FloatTensor([0.0, 1.0]).to('cuda'))
    
        G.train(D, titles, torch.FloatTensor([1.0, 0.0]).to('cuda'))

    D.plot_progress()
    
    
def train_one_epoch_gpt2(loader, D, G, D_optimizer_, D_scheduler_, G_optimizer_, G_scheduler_, device_, writer):
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
        real_labels = {'labels':torch.stack([torch.tensor([0,1], dtype=torch.float).to(device_) for i in range(len(titles))])}
        fake_labels = {'labels':torch.stack([torch.tensor([1,0], dtype=torch.float).to(device_) for i in range(len(titles))])}
        #fake_labels = {'labels':torch.tensor([0 for i in range(len(titles))], dtype=torch.long).to(device_)}
        real_input.update(real_labels)
        
        #print(D.model.config)
        D.train(real_input, D_optimizer_, D_scheduler_)
        generator_output = G.model(input_ids = generators_input['input_ids'].detach(), attention_mask=generators_input['attention_mask'].detach())
        generator_probs = torch.softmax(generator_output[0], dim=-1).squeeze()
        fake_abstracts_ids = [torch.multinomial(generator_probs[i], num_samples=1).squeeze() for i in range(len(generator_probs))]
        decoded = [G.tokenizer.decode(ids) for ids in fake_abstracts_ids]
        fake_input_list = [D.tokenizer(item, padding="max_length", truncation=True, return_tensors='pt', max_length=D.tokenizer.model_max_length) for item in decoded]
        fake_input = {}
        for item in fake_input_list:
            for k, v in item.items():
                if k not in fake_input.keys():
                    fake_input[k] = []
                fake_input[k].append(v)
        fake_input = {k:torch.stack(v).squeeze() for k,v in fake_input.items()}
        #fake_abstracts = [G.model.generate(generators_input['input_ids'].detach()).squeeze() for i in range(len(generators_input['input_ids']))]
        #print('FAKE ABSTRACTS', fake_abstracts)
        #fake_abstracts_tokenized = [D.tokenizer(G.tokenizer.decode(fake_abstracts[i]), padding="max_length", truncation=True, return_tensors='pt', max_length=512) for i in range(len(fake_abstracts))]
        #fake_abstracts_input_ids_stacked = torch.stack([fake_abstracts_tokenized[i]['input_ids'] for i in range(len(fake_abstracts_tokenized))],dim=1).squeeze()
        #fake_abstracts_attention_mask_stacked = torch.stack([fake_abstracts_tokenized[i]['attention_mask'] for i in range(len(fake_abstracts_tokenized))],dim=1).squeeze()

        #fake_input = {'input_ids':fake_abstracts_input_ids_stacked, 'attention_mask':fake_abstracts_attention_mask_stacked}
        fake_input.update(fake_labels)
        fake_input = {k:v.to(device_) for k,v in fake_input.items()}
        
        D.train(fake_input, D_optimizer_, D_scheduler_)
        
        G.train(D, generators_input, G_optimizer_, G_scheduler_)
        
        
        print("Generator Loss: "+str(G.running_loss/(i+1)))
        print("Discriminator Loss: "+str(D.running_loss/(i+1)))
        wandb.log({"gen_train_loss":G.running_loss/(i+1), "disc_train_loss":D.running_loss/(i+1)})
        writer.add_scalars('Running avg loss',
                            { 'Generator' : G.running_loss/(i+1), 
                            'Discriminator': D.running_loss/(i+1)
                            },
                            i)
        writer.flush()
    D.running_loss = 0
    G.running_loss = 0
        
        
        
    
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
    
    train_dataset = load_from_disk('data/train_tokenized_json_small_double_gpt')
    valid_dataset = load_from_disk('data/valid_tokenized_json_small_double_gpt')
    test_dataset = load_from_disk('data/test_tokenized_json_small_double_gpt')
    
    train_loader = DataLoader(train_dataset['train'], num_workers=0, collate_fn=collate_fn_GPT2, batch_size=1)
    tokenizer = AutoTokenizer.from_pretrained("./gpt2_tokenizer")
    #tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = "left"
    
    
    D = GPT2Discriminator('gpt2', tokenizer=tokenizer)
    
    tokenizer = AutoTokenizer.from_pretrained('./gpt2_tokenizer')
    #tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    G = GPT2Generator('gpt2-finetuning_again_new_better/checkpoint-2000', tokenizer=tokenizer)
    
    D_optimizer_ = AdamW(D.model.parameters(),
                  lr = 5e-4, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )
    
    G_optimizer_ = AdamW(D.model.parameters(),
                  lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(train_loader) * EPOCHS

    # Create the learning rate scheduler.
    D_scheduler_ = get_linear_schedule_with_warmup(D_optimizer_,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    G_scheduler_ = get_linear_schedule_with_warmup(G_optimizer_,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/gan{}'.format(timestamp))
    
    train_one_epoch_gpt2(train_loader, D, G, D_optimizer_, D_scheduler_, G_optimizer_, G_scheduler_, 'cuda', writer)
    
    
    
    
    

    
    
    
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
    