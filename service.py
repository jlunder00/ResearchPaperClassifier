import bentoml
import sys, os, json
import transformers
from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch



svc = bentoml.Service("papers_classifier") 

model_path = "discriminator/model/"
tokenizer_path = "disciminator/tokenizer/"

discriminator_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
discriminator_config = GPT2Config.from_pretrained(model_path, num_labels=2)
discriminator_config.vocab_size = len(discriminator_tokenizer)
discriminator_config.pad_token_id = discriminator_tokenizer.pad_token_id
discriminator_config.pad_token = discriminator_tokenizer.pad_token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator_model = GPT2ForSequenceClassification.from_pretrained(model_path, config=discriminator_config, ignore_mismatched_sizes=True)
discriminator_model.to(device)




@svc.api(input=JSON(), output=JSON())
def predict(input:JSON):
    '''
    Take in an abstract and predict whether it is AI generated or not
    '''
    input_text = input['text']
    tokenized_input = discriminator_tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    output = discriminator_model(tokenized_input)
    prediction = torch.argmax(output[0])
    return {'prediction':prediction}

    
    

