#decoder layer
#general decoder = n*decoder layer
import torch
import torch.nn as nn
from helpers import clones
from MultiHeadAttn import MultiHeadAttention
from FeedForward import FeedForwardLayer

class Decoder(nn.Module):
    def __init__(self,layers,decoderlayer):
        self.layers=layers
        self.decoder = decoderlayer
        self.decoderlayers = [clones(self.decoder,self.layers)]
    
    def forward(self,encoder_outputs,x):
        for decodinglayer in self.decoderlayers:
            x = decodinglayer(x,encoder_outputs)
        return x
    

class LayerDecoder(nn.Module):
    def __init__(self,heads,dmodel):
        super().__init__()
        self.heads=heads
        self.dmodel=dmodel
        self.norm = nn.LayerNorm(self.dmodel)
        self.mham = MultiHeadAttention(self.dmodel,True,self.heads)
        self.mha = MultiHeadAttention(self.dmodel,False,self.heads)
        self.ff = FeedForwardLayer(2048,self.dmodel)
        self.lin = nn.Linear(self.dmodel,2*self.dmodel)
        self.lin2 = nn.Linear(self.dmodel,3*self.dmodel)
        self.lin3 = nn.Linear(self.dmodel,self.dmodel)
    
    def forward(self,x,encoder_output):
        K,V = self.lin(encoder_output).chunk(2,-1)
        x = self.lin3(x)
        masked_attention =self.norm(x+self.mham(x,K,V))
        newQ,newK,newV = self.lin2(masked_attention).chunk(3,-1)
        unmasked_attention = self.norm(self.mha(newQ,newK,newV)+masked_attention)
        output = self.norm(self.ff(unmasked_attention)+unmasked_attention)
        return output

masked = torch.randn((5,5,512))
encoder = torch.randn((5,5,512)) 
thing = LayerDecoder(8,512)
print(thing.forward(masked,encoder))