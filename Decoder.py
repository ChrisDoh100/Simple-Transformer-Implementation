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
    """Individual decoder Layer, consisting of multi-head attention layer and feedforward layer."""
    def __init__(self,config):
        super(LayerDecoder,self).__init__()
        self.heads = config['heads']
        self.dmodel = config['model_dimension']
        self.dropout_prob  = config['dropout_prob']
        
        # Linear layers for transforming inputs and encoder output
        
        # Multi-head attention layers for self-attention and encoder-decoder attention
        self.mham = MultiHeadAttention(config)  # Masked self-attention
        self.mha = MultiHeadAttention(config)   # Encoder-decoder attention
        
        # Separate LayerNorm instances for each attention and feed-forward submodule
        self.norm1 = nn.LayerNorm(self.dmodel)  # For masked self-attention
        self.norm2 = nn.LayerNorm(self.dmodel)  # For encoder-decoder attention
        self.norm3 = nn.LayerNorm(self.dmodel)  # For feed-forward layer
        
        # Feed-forward network
        self.ff = FeedForwardLayer(config)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.dropout1= nn.Dropout(p=self.dropout_prob)
        self.dropout2 = nn.Dropout(p=self.dropout_prob)

    def forward(self, trg, encoder_output, trg_mask,src_mask):
        #We use a pre-norm implementation, mostly due to the fact that we are using a low data regime.
        trb = self.norm1(trg)
        attention = self.mham(query_matrix=trb, key_matrix=trb, value_matrix=trb, mask=trg_mask)
        #Decoder Block 1
        unmasked_attention = trg + self.dropout(attention)
        #Decoder Block 2
        normed_attention = self.norm2(unmasked_attention)
        cross_attention = self.mha(query_matrix=normed_attention, key_matrix=encoder_output, value_matrix=encoder_output, mask=src_mask)
        masked_attention = unmasked_attention + self.dropout1(cross_attention)
        #Decoder Block 3
        normed_masked_attention = self.norm3(masked_attention)
        feedforward = self.ff(normed_masked_attention)
        output = masked_attention + self.dropout2(feedforward)

        return output
