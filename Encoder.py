#encoder layer
#general encoder = n*encoder layer
from torch import nn
from MultiHeadAttn import MultiHeadAttention
from FeedForward import FeedForwardLayer
from Helpers import clones

class Encoder(nn.Module):
    """Encoder Module of the Transformer"""
    def __init__(self,layers,encoder):
        super().__init__()
        assert isinstance(encoder,LayerEncoder)
        self.layers=layers
        self.encoder = encoder
        self.encoderlayers = clones(self.encoder,self.layers)
        self.norm = nn.LayerNorm(encoder.dmodel)

    def forward(self,x,mask):
        """Forward pass through all the encoder layers, depending
            on how many were specified."""
        src_tokens=x
        for encodinglayer in self.encoderlayers:
            src_tokens = encodinglayer(src_tokens,mask)
        return self.norm(src_tokens)

class LayerEncoder(nn.Module):
    """This is a single encoder layer.(no shit captain obvious....)"""

    def __init__(self, config):
        super().__init__()
        self.heads = config['heads']
        self.dmodel = config['model_dimension']
        self.dropout_prob = config['dropout_prob']
        
        # Multi-head attention layer
        self.mha = MultiHeadAttention(config)
        
        # Separate LayerNorm instances for attention and feed-forward parts
        self.norm1 = nn.LayerNorm(self.dmodel)  # For attention submodule
        self.norm2 = nn.LayerNorm(self.dmodel)  # For feed-forward submodule
        
        # Feed-forward layer
        self.ff = FeedForwardLayer(config)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(p=self.dropout_prob)#Multi-Head Attention Dropout
        self.dropout2 = nn.Dropout(p=self.dropout_prob)#Feedforward Dropout

    def forward(self, trg_batch, src_mask):
        """Forward pass through a specific encoder layer."""
        #For each sub layer we use X + sublayer(norm(x)) instead of norm(X+sublayer(x)), explained in the readme.
        #We apply dropout to the output of each sublayer as a regularization technique but its not very important in terms of the overall architecture.
        #Encoder Sub-Block 1
        trb = self.norm1(trg_batch) #  = norm(x)
        out_encoder = trg_batch + self.dropout1(self.mha(trb,trb,trb,mask=src_mask)) # = X+Multi-Head Attention(norm(x))
        #Encoder Sub-Block 2
        normed_out_encoder = self.norm2(out_encoder) # = norm(x)
        output = out_encoder + self.dropout2(self.ff(normed_out_encoder))# = X+Feedforward(norm(x))
        return output

