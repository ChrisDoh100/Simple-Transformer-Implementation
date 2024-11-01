import copy
from torch import nn


def clones(module,N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def num_params(model):
    """Returns the total number of tunable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def type_params(model):
    for name,p in model.named_parameters(): 
        print(name,p.dtype,'\n')