import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOutputGenerator(nn.Module):
    
    def __init__(self,dmodel,vocab_size):
        super().__init__()
        self.lin = nn.Linear(dmodel,vocab_size)
    
    def forward(self,x):
        return F.softmax(self.lin(x),dim=-1)