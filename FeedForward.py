import torch
import torch.nn as nn



class FeedForwardLayer(nn.Module):
    def __init__(self,inter,output):
        super().__init__()
        self.output = output
        self.intermediate = inter
        self.firstfeedforward = nn.Linear(self.output,self.intermediate)
        self.secondfeedfoward = nn.Linear(self.intermediate,self.output)
        self.relu = nn.ReLU()
    

    def forward(self,x):
        x  = self.firstfeedforward(x)
        x  = self.relu(x)
        x  = self.secondfeedfoward(x)
        return x
    
