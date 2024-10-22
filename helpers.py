import copy
from torch import nn
from Config import config
from datasets import load_dataset


def clones(module,N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_data(basetxtfile, transtxtfile,valid=False):
    """Retrieves the data and then clips it to the desired amount of setences."""

    firstland = [line.strip('\n')  for line in open(basetxtfile,encoding='utf-8')]
    secondland = [line.strip('\n') for line in open(transtxtfile,encoding='utf-8')]
    if valid is False:
        firstland = firstland[:config['sentence_amount']]
        secondland = secondland[:config['sentence_amount']]
    else:
        firstland = firstland[:config['val_sentence_amount']]
        secondland = secondland[:config['val_sentence_amount']]
    return firstland,secondland



def num_params(model):
    """Returns the total number of tunable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def type_params(model):
    for name,p in model.named_parameters(): 
        print(name,p.dtype,'\n')