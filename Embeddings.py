from torch import nn
import sentencepiece as spm
import math


class Embeddings(nn.Module):
    """Generates the embeddings for each input sentence/matrix"""
    def __init__(self,config):
        super().__init__()
        self.dim_model=config['model_dimension']
        self.embed = nn.Embedding(config['vocab_size'],config['model_dimension'],padding_idx=3,norm_type=2)

    def forward(self,tokens):
        """Forward pass for embeddings module, returns a set of embeddings
            scaled by the square root of the models dimension/embedding dimension."""
        embeddings = self.embed(tokens)
        scaling_factor = math.sqrt(self.dim_model)
        embeddings = embeddings * scaling_factor
        return embeddings