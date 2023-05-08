import json
from tqdm import tqdm




def load(fname):
    with open(fname, 'r') as fin:
        data = json.load(fin)
    return data

def write(fname, data):
    with open(fname, 'w') as fout:
        for item in data:
            line = data['title']+': '+data['abstract']
            fout.write(line+'\n')
