import bentoml
from bentoml.io import JSON, Text
import sys, os, json
import transformers
from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch




model_path = "discriminator/model"
tokenizer_path = "discriminator/tokenizer"

discriminator_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
discriminator_config = GPT2Config.from_pretrained(model_path, num_labels=2)
discriminator_config.vocab_size = len(discriminator_tokenizer)
discriminator_config.pad_token_id = discriminator_tokenizer.pad_token_id
discriminator_config.pad_token = discriminator_tokenizer.pad_token
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bentoml.pytorch.save("discriminator_model", GPT2ForSequenceClassification.from_pretrained(model_path, config=discriminator_config, ignore_mismatched_sizes=True))
#print(discriminator_model.device)

#bentoml.transformers.save_model("discriminator_model", discriminator_model)


#discriminator_runner = bentoml.transformers.get("discriminator_model").to_runner()
discriminator_runner = bentoml.pytorch.get("discriminator_model").to_runner()


svc = bentoml.Service("papers_classifier", runners=[discriminator_runner]) 


@svc.api(input=Text(), output=JSON())
def predict(input:str):
    '''
    Take in an abstract and predict whether it is AI generated or not
    '''
    input_text = input
    tokenized_input = discriminator_tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    output = discriminator_runner.run(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])
    prediction = torch.argmax(output[0])
    print(prediction)

    return {'prediction':prediction.item()}

    
    

