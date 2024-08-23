from torch import nn
import torch.nn.functional as F

class DecoderOutputGenerator(nn.Module):
    """Final module for the transformer that generates our output."""
    def __init__(self,dmodel,vocab_size):
        super().__init__()
        self.lin = nn.Linear(dmodel,vocab_size)

    def forward(self,x):
        """Forward pass that generates the final output."""
        return F.softmax(self.lin(x),dim=-1)
