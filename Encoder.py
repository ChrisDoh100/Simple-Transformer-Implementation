#encoder layer
#general encoder = n*encoder layer
import torch
import torch.nn as nn
from MultiHeadAttn import MultiHeadAttention
from FeedForward import FeedForwardLayer
from helpers import clones

class Encoder(nn.Module):
    def __init__(self,layers,encoder):
        super().__init__()
        self.layers=layers
        self.encoder = encoder
        self.encoderlayers = clones(self.encoder,self.layers)
        self.norm = nn.LayerNorm(self.encoder.dmodel)
    
    def forward(self,x):
        for encodinglayer in self.encoderlayers:
            x = encodinglayer(x)
        return x
    




class LayerEncoder(nn.Module):
    def __init__(self,heads,dmodel):
        super().__init__()
        self.heads = heads
        self.dmodel = dmodel
        self.mha = MultiHeadAttention(self.dmodel,self.heads)
        self.ff = FeedForwardLayer(2048,self.dmodel)
        self.lin = nn.Linear(self.dmodel,3*self.dmodel)
    
    def forward(self,x):
        input = x
        Q,K,V =self.lin(x).chunk(3,-1) 
        x = self.mha(Q,K,V)
        outmha = input+x
        norm = nn.LayerNorm(outmha.shape)
        outmha = norm(outmha)
        outencoder = norm(self.ff(outmha)+outmha)
        return outencoder


thing = torch.randn((5,5,10),requires_grad=True)
other = LayerEncoder(2,10,True)
print(other.forward(thing))