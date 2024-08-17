import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOutputGenerator(nn.Module):
    
    def __init__(self,dmodel,vocab_size):
        super().__init__()
        self.lin = nn.Linear(dmodel,vocab_size)
        self.softmax = F.softmax(dim=-1)
    
    def forward(self,x):
        return self.softmax(self.lin(x))