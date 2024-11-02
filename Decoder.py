#decoder layer
#general decoder = n*decoder layer
import torch
from torch import nn
from Helpers import clones
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
        #We use a pre-norm implementation,explained in the readme.
        #We apply dropout to the output of each sublayer as a regularization technique but its not very important in terms of the overall architecture.

        #Decoder Block 1
        # X+Multi-Head Masked Attention(norm(X))

        trb = self.norm1(trg) #=norm(X)
        attention = self.mham(query_matrix=trb, key_matrix=trb, value_matrix=trb, mask=trg_mask) # = Multi-Head Masked Attention(norm(X))
        unmasked_attention = trg + self.dropout(attention)# = X+Multi-Head Masked Attention(norm(X))

        #Decoder Block 2
        # X+Multi-Head Attention(norm(X))

        normed_unmasked_attention = self.norm2(unmasked_attention) #=norm(X)
        cross_attention = self.mha(query_matrix=normed_unmasked_attention, key_matrix=encoder_output, value_matrix=encoder_output, mask=src_mask)#= Multi-Headed Attention(norm(X))
        masked_attention = unmasked_attention + self.dropout1(cross_attention)# = X+Multi-Head Attention(norm(X))

        #Decoder Block 3
        # X+FeedForward(norm(X))
        normed_masked_attention = self.norm3(masked_attention)#=norm(X)
        feedforward = self.ff(normed_masked_attention)# = FeedForward(norm(X))
        output = masked_attention + self.dropout2(feedforward)# = X+FeedForward(norm(X))

        return output
