from torch import nn
import sentencepiece as spm
import math


sp = spm.SentencePieceProcessor()
try:
    sp.Load('training.model')
except:
    print("w/e")

class Embeddings(nn.Module):
    """Generates the embeddings for each input sentence/matrix"""
    def __init__(self,vocab_size,dim_model):
        super().__init__()
        self.dim_model=dim_model
        self.embed = nn.Embedding(vocab_size,dim_model,padding_idx=sp.pad_id())

    def forward(self,tokens):
        """Forward pass for embeddings module, returns a set of embeddings
            scaled by the square root of the models dimension/embedding dimension."""
        embeddings = self.embed(tokens)
        scaling_factor = math.sqrt(self.dim_model)
        embeddings = embeddings * scaling_factor
        return embeddings