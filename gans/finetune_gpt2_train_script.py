from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from datasets import load_dataset, load_from_disk



def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_path = 'data/gpt2_train.txt'
    val_path = 'data/gpt2_valid.txt'
    test_path = 'data/gpt2_test.txt'
    tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"



    train_dataset = load_from_disk("data/train_tokenized_json_small")
    val_dataset = load_from_disk("data/valid_tokenized_json_small")


    val_dataset = val_dataset['train']
    train_dataset = train_dataset['train']


    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
    

    model = AutoModelWithLMHead.from_pretrained("gpt2")




    training_args = TrainingArguments(
        output_dir="./gpt2-finetuning_again_best", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=3, # number of training epochs
        per_device_train_batch_size=4, # batch size for training
        per_device_eval_batch_size=4,  # batch size for evaluation
        eval_steps = 200, # Number of update steps between two evaluations.
        save_steps= 500, # after # steps model is saved
        warmup_steps=500,# number of warmup steps for learning rate scheduler
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    trainer.save_model()
    
if __name__ == "__main__":
    main()