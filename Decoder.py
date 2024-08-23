#decoder layer
#general decoder = n*decoder layer
from torch import nn
from helpers import clones
from MultiHeadAttn import MultiHeadAttention
from FeedForward import FeedForwardLayer

class Decoder(nn.Module):
    """Decoder Module for the Transformer."""

    def __init__(self,layers,decoderlayer):
        super().__init__()
        self.layers=layers
        self.decoder = decoderlayer
        self.decoderlayers = clones(self.decoder,self.layers)

    def forward(self,encoder_outputs,x,fr_att_mask,fr_pad_msk):
        for decodinglayer in self.decoderlayers:
            x = decodinglayer(x,encoder_outputs,fr_att_mask,fr_pad_msk)
        return x
    

class LayerDecoder(nn.Module):
    def __init__(self,heads,dmodel):
        super().__init__()
        self.heads=heads
        self.dmodel=dmodel
        self.lin2 = nn.Linear(self.dmodel,3*self.dmodel)
        self.mham = MultiHeadAttention(dmodel=self.dmodel,heads=self.heads)
        self.norm = nn.LayerNorm(self.dmodel)
        self.ff = FeedForwardLayer(2048,self.dmodel)
        self.lin = nn.Linear(self.dmodel,2*self.dmodel)
        self.mha = MultiHeadAttention(dmodel=self.dmodel,heads=self.heads)

    def forward(self,x,encoder_output,fr_attn_mask,fr_padding_msk):
        k_matrix,v_matrix = self.lin(encoder_output).chunk(2,-1)
        masked_attention =self.norm(x+self.mham(x,k_matrix,v_matrix,fr_attn_mask))
        newq_matrix,newk_matrix,newv_matrix = self.lin2(masked_attention).chunk(3,-1)
        unmasked_attention = self.norm(self.mha(newq_matrix,newk_matrix,newv_matrix,fr_padding_msk)+masked_attention)
        output = self.norm(self.ff(unmasked_attention)+unmasked_attention)
        return output
