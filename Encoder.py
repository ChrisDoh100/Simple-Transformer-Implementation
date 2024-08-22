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
    
    def forward(self,x,mask):
        for encodinglayer in self.encoderlayers:
            x = encodinglayer(x,mask)
        return x
    

class LayerEncoder(nn.Module):
    def __init__(self,heads,dmodel):
        super().__init__()
        self.heads = heads
        self.dmodel = dmodel
        self.lin = nn.Linear(self.dmodel,3*self.dmodel)
        self.mha = MultiHeadAttention(self.dmodel,self.heads)
        self.norm = nn.LayerNorm(self.dmodel)
        self.ff = FeedForwardLayer(2048,self.dmodel)
    
    def forward(self,x,mask):
        input = x
        Q,K,V =self.lin(x).chunk(3,-1)
        x = self.mha(Q,K,V,mask)
        outmha = input+x
        outmha = self.norm(outmha)
        outencoder = self.norm(self.ff(outmha)+outmha)
        return outencoder
