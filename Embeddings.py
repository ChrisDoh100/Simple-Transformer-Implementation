import math
from torch import nn




class Embeddings(nn.Module):
    """Generates the embeddings for each input sentence/matrix"""
    def __init__(self,vocab_size,dim_model):
        super().__init__()
        self.dim_model = dim_model
        self.voc_size=vocab_size
        self.embed = nn.Embedding(self.voc_size,self.dim_model)

    def forward(self,tokens):
        """Forward pass for embeddings module, returns a set of embeddings
            scaled by the square root of the models dimension/embedding dimension."""
        embeddings = self.embed(tokens)
        embeddings = embeddings*math.sqrt(self.dim_model)
        return embeddings