import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)

def scaledDotProduct(Q,K,V,dim_model,mask,heads):
        Q = Q.reshape(Q.shape[0],heads,Q.shape[1],Q.shape[-1]//heads)
        K = K.reshape(K.shape[0],heads,K.shape[1],K.shape[-1]//heads)
        V = V.reshape(V.shape[0],heads,V.shape[1],V.shape[-1]//heads)
        KT = torch.transpose(K,-2,-1)
        attention = torch.matmul(Q,KT)
        attention = torch.div(attention,math.sqrt(dim_model//heads))
        if mask is not False:
            attention = torch.tril(attention)
            attention = attention.masked_fill(attention==0,float("-inf"))
        scaled = F.softmax(attention,dim=3)
        values = torch.matmul(scaled,V)
        return values 


class MultiHeadAttention(nn.Module):
    def __init__(self,dmodel,heads,mask=False):
        super().__init__()
        self.dmodel = dmodel
        self.mask = mask
        self.heads = heads
        self.headdim = dmodel//heads
        self.lin = nn.Linear(self.dmodel,self.dmodel)
    

    def forward(self,Q,K,V):
         x = scaledDotProduct(Q,K,V,self.dmodel,self.mask,self.heads)
         x = x.view(x.size(dim=0),x.size(dim=2),self.headdim*self.heads)
         x = self.lin(x)
         return x
