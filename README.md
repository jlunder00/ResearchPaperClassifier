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

