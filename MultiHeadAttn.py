import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)

def scaled_dot_product(query_matrix,key_matrix,value_matrix,dim_model,heads,mask=None):
    """Computes the Attenction value outputs."""

    query_heads_matrix = query_matrix.reshape(query_matrix.shape[0],heads,query_matrix.shape[1],query_matrix.shape[-1]//heads)
    key_heads_matrix = key_matrix.reshape(key_matrix.shape[0],heads,key_matrix.shape[1],key_matrix.shape[-1]//heads)
    value_heads_matrix = value_matrix.reshape(value_matrix.shape[0],heads,value_matrix.shape[1],value_matrix.shape[-1]//heads)
    key_transpose = torch.transpose(key_heads_matrix,-2,-1)
    attention = torch.matmul(query_heads_matrix,key_transpose)
    attention = torch.div(attention,math.sqrt(dim_model//heads))
    if mask is not None:
        attention+=mask
    scaled = F.softmax(attention,dim=3)
    values = torch.matmul(scaled,value_heads_matrix)
    return values


class MultiHeadAttention(nn.Module):
    """Computes the Multi Head attention aspect of the Transformer."""
    def __init__(self,dmodel,heads):
        super().__init__()
        self.dmodel = dmodel
        self.heads = heads
        self.headdim = dmodel/heads
        self.lin = nn.Linear(self.dmodel,self.dmodel)


    def forward(self,query_matrix,key_matrix,value_matrix,mask):
        """Forward pass of the Multi-Headed Attention."""

        x = scaled_dot_product(query_matrix,key_matrix,value_matrix,dim_model=self.dmodel,mask=mask,heads=self.heads)
        x = x.view(x.size(dim=0),x.size(dim=2),self.dmodel)
        x = self.lin(x)
        return x
