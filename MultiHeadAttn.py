import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)


class MultiHeadAttention(nn.Module):
    """Computes the Multi Head attention aspect of the Transformer."""
    def __init__(self,dmodel,heads):
        super().__init__()
        self.dmodel = dmodel
        self.heads = heads
        self.headdim = int(dmodel/heads)
        self.lin = nn.Linear(self.dmodel,self.dmodel)
        self.lin1 = nn.Linear(self.dmodel,self.dmodel)
        self.lin2 = nn.Linear(self.dmodel,self.dmodel)
        self.lin3=nn.Linear(self.dmodel,self.dmodel)
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
        query_matrix = self.lin(query_matrix).view(query_matrix.shape[0],-1,self.heads,self.headdim).transpose(1,2)
        
        key_matrix = self.lin1(key_matrix).view(key_matrix.shape[0],-1,self.heads,self.headdim).transpose(1,2)

        value_matrix = self.lin2(value_matrix).view(value_matrix.shape[0],-1,self.heads,self.headdim).transpose(1,2)
        x = self.scaled_dot_product(query_matrix,key_matrix,value_matrix,mask=mask)
        x = x.transpose(1,2)
        x=x.reshape(x.shape[0],-1,self.dmodel)
        x = self.lin3(x)
        return x
