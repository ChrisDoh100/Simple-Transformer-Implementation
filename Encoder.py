#encoder layer
#general encoder = n*encoder layer
from torch import nn
from MultiHeadAttn import MultiHeadAttention
from FeedForward import FeedForwardLayer
from helpers import clones

class Encoder(nn.Module):
    """Encoder Module of the Transformer"""
    def __init__(self,layers,encoder):
        super().__init__()
        self.layers=layers
        self.encoder = encoder
        self.encoderlayers = clones(self.encoder,self.layers)

    def forward(self,x,mask):
        """Forward pass through all the encoder layers, depending
            on how many were specified."""
        for encodinglayer in self.encoderlayers:
            x = encodinglayer(x,mask)
        return x

class LayerEncoder(nn.Module):
    """This is a single encoder layer."""

    def __init__(self,heads,dmodel):
        super().__init__()
        self.heads = heads
        self.dmodel = dmodel
        self.lin = nn.Linear(self.dmodel,3*self.dmodel)
        self.mha = MultiHeadAttention(self.dmodel,self.heads)
        self.norm = nn.LayerNorm(self.dmodel)
        self.ff = FeedForwardLayer(2048,self.dmodel)

    def forward(self,x,mask):
        """Forward pass through a specific layer, returns
            output that will be passed to decoder."""
        initial_input = x
        q_matrix,k_matrix,v_matrix =self.lin(x).chunk(3,-1)
        x = self.mha(q_matrix,k_matrix,v_matrix,mask)
        outmha = x+initial_input
        outmha = self.norm(outmha)
        outencoder = self.norm(self.ff(outmha)+outmha)
        return outencoder
