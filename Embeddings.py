import torch
import torch.nn as nn
import math



class Embeddings(nn.Module):
    def __init__(self,vocab_size,dim_model):
        super().__init__()
        self.dim_model = dim_model
        self.voc_size=vocab_size
        self.embed = nn.Embedding(self.voc_size,self.dim_model)

    def forward(self,tokens):
        embeddings = self.embed(tokens)
        embeddings = embeddings*math.sqrt(self.dim_model)
        return embeddings