# ResearchPaperClassifier
A classification algorithm for identifying properties about a scientific paper from its text, particularly whether it was published via predatory/non reputable publisher


# Prework for tutorial:
1. Create a python virtual environment: `python -m venv path/to/venv/folder`
1. Activate the environment (dependent on shell/system). In bash on linux: `source path/to/venv/folder/bin/activate`
1. Clone and enter this repo
1. Install the requirements: `pip install -r requirements.txt` (this may take awhile because of pytorch)
1. Run `wandb login`
1. Follow the instructions to create a profile if you don't have one, then get and paste your api key
NOTE: If you don't have a decent GPU with cuda installed, it will use CPU to train, and that will be very, very slow.    
Run `nvidia-smi` to check 

# How to use
go to http://73.254.3.61:5003/ to use the classifier by providing an abstract, receiving a prediction on whether it is fake or real

## How it works
The discriminator is trained using a Generative Adversarial Network architecture using a finetuned pretrained GPT2 as the Generator and an non finetuned pretrained GPT2 as the discriminator.    

It does not work very well at the moment. More computational power and training time is needed

