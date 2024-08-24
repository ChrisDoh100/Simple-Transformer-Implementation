import copy
from torch import nn
from Config import config


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
    return firstland,secondland

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)