#positional encoding formula
import torch
from torch import nn
#PE(pos,2i) = sin(pos/10000**(2i/dmodel))
#PE(pos,2i+1) = cos(pos/10000**(2i/dmodel))

#these positional encodings are added to the input embeddings but are not learnable

class PositionalEncodings(nn.Module):
    """Positional encodings that is added to the initial encodings
        and passed the the encoder/decoder."""
    def __init__(self,config,sequence_length=5000):
        super().__init__()
        self.max_sec_len = sequence_length
        self.model_dimension=config['model_dimension']
        self.dropout_prob=config['dropout_prob']
        self.dropout = nn.Dropout(p=self.dropout_prob)
        #math for encodings
        positions = torch.arange(0,self.max_sec_len).unsqueeze(1)
        freq = torch.pow(10000.,-torch.arange(0,self.model_dimension,2,dtype=torch.float)/self.model_dimension)

        #creating positional encodings table
        pos_embed = torch.zeros(self.max_sec_len,self.model_dimension)
        #calculating the embeddings for every even pos.
        pos_embed[:,0::2] = torch.sin(positions*freq)
        #calculating the embeddings for every odd pos.
        pos_embed[:,1::2] = torch.cos(positions*freq)
        #adding positional encodings to the state buffer to so that it can be moved to gpu
        #but not treated as a parameter to do backprop on.
        self.register_buffer('pos_embed',pos_embed)


    

    def forward(self,input_embeddings):
        """Gets the embeddings for the given sequence based on the 
            given length of the sequence."""
        #getting the amount of rows from out positional encoding we need.
        #extracting them from our main positional encodings.
        pos_embed_porition = self.pos_embed[:input_embeddings.shape[1]]
        
        return self.dropout(input_embeddings+pos_embed_porition)
