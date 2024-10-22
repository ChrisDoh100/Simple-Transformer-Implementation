#decoder layer
#general decoder = n*decoder layer
import torch
from torch import nn
from helpers import clones
from MultiHeadAttn import MultiHeadAttention
from FeedForward import FeedForwardLayer

class Decoder(nn.Module):
    """Decoder Module for the Transformer."""

    def __init__(self,layers,decoderlayer):
        super().__init__()
        self.decoderlayers = clones(decoderlayer,layers)
        self.norm = nn.LayerNorm(decoderlayer.dmodel)

    def forward(self,x,encoder_outputs,trg_mask,src_mask):
        trg_tokens=x
        for decodinglayer in self.decoderlayers:
            trg_tokens = decodinglayer(trg_tokens,encoder_outputs,trg_mask,src_mask)
        return self.norm(trg_tokens)
    

class LayerDecoder(nn.Module):
    def __init__(self, heads, dmodel):
        super(LayerDecoder,self).__init__()
        self.heads = heads
        self.dmodel = dmodel
        
        # Linear layers for transforming inputs and encoder output
        
        # Multi-head attention layers for self-attention and encoder-decoder attention
        self.mham = MultiHeadAttention(dmodel=self.dmodel, heads=self.heads)  # Masked self-attention
        self.mha = MultiHeadAttention(dmodel=self.dmodel, heads=self.heads)   # Encoder-decoder attention
        
        # Separate LayerNorm instances for each attention and feed-forward submodule
        self.norm1 = nn.LayerNorm(self.dmodel)  # For masked self-attention
        self.norm2 = nn.LayerNorm(self.dmodel)  # For encoder-decoder attention
        self.norm3 = nn.LayerNorm(self.dmodel)  # For feed-forward layer
        self.norm = nn.LayerNorm(self.dmodel)
        self.norm4 = nn.LayerNorm(self.dmodel)
        
        # Feed-forward network
        self.ff = FeedForwardLayer(2048, self.dmodel)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        self.dropout1= nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, trg, encoder_output, trg_mask,src_mask):
        #Normalizing Inputs, for some reason the model has a hard time learning if you don't normalise the initial inputs.
        trb = self.norm(trg)
        x = self.mham(query_matrix=trb, key_matrix=trb, value_matrix=trb, mask=trg_mask)
        #Decoder Block 1
        unmasked_attention = trg + self.dropout(self.norm1(x))
        #Decoder Block 2
        masked_attention = unmasked_attention + self.dropout1(self.mha(query_matrix=self.norm2(unmasked_attention), key_matrix=encoder_output, value_matrix=encoder_output, mask=src_mask))
        #Decoder Block 3
        output = masked_attention + self.dropout2(self.ff(self.norm3(masked_attention)))

        return output
