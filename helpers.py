import copy
from torch import nn


def clones(module,N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_data(basetxtfile, transtxtfile,kwargs):
    """Retrieves the data and then clips it to the desired amount of setences."""

    firstland = [line.strip('\n')  for line in open(basetxtfile,encoding='utf-8')]
    secondland = [line.strip('\n') for line in open(transtxtfile,encoding='utf-8')]
    firstland = firstland[:kwargs['sentence_amount']]
    secondland = secondland[:kwargs['sentence_amount']]
    return firstland,secondland
