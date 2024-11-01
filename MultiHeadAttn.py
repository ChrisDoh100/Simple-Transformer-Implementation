import math
import torch
from torch import nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    """Computes the Multi Head attention aspect of the Transformer."""
    def __init__(self,config):
        super().__init__()
        self.dmodel = config['model_dimension']
        self.heads = config['heads']
        self.headdim = int(self.dmodel/self.heads)
        self.query_transform = nn.Linear(self.dmodel,self.dmodel)
        self.key_transform= nn.Linear(self.dmodel,self.dmodel)
        self.value_transform = nn.Linear(self.dmodel,self.dmodel)
        self.output_transform=nn.Linear(self.dmodel,self.dmodel)
        self.softmax = nn.Softmax(dim=-1)

    def scaled_dot_product(self,query_matrix,key_matrix,value_matrix,mask=None):
        """Computes the Attention value outputs."""
        key_transpose = key_matrix.transpose(-2,-1)
        attention = torch.matmul(query_matrix,key_transpose)
        attention = attention/math.sqrt(self.headdim)
        if mask is not None:
            attention.masked_fill_(mask == torch.tensor(False), float("-inf"))
        scaled = self.softmax(attention)
        values = torch.matmul(scaled,value_matrix)
        return values

    def forward(self,query_matrix,key_matrix,value_matrix,mask):
        """Forward pass of the Multi-Headed Attention."""
        query_matrix = self.query_transform(query_matrix).view(query_matrix.shape[0],-1,self.heads,self.headdim).transpose(1,2)
        
        key_matrix = self.key_transform(key_matrix).view(key_matrix.shape[0],-1,self.heads,self.headdim).transpose(1,2)

        value_matrix = self.value_transform(value_matrix).view(value_matrix.shape[0],-1,self.heads,self.headdim).transpose(1,2)
        dot_product_output = self.scaled_dot_product(query_matrix,key_matrix,value_matrix,mask=mask)
        dot_product_output = dot_product_output.transpose(1,2)
        dot_product_output=dot_product_output.reshape(dot_product_output.shape[0],-1,self.dmodel)
        mha_output = self.output_transform(dot_product_output)
        return mha_output
