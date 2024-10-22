from torch import nn

class DecoderOutputGenerator(nn.Module):
    """Final module for the transformer that generates our output."""
    def __init__(self,dmodel,vocab_size):
        super().__init__()
        self.lin = nn.Linear(dmodel,vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.vocab_size = vocab_size
    def forward(self,x):
        """Forward pass that generates the final output."""
        return self.logsoftmax(self.lin(x)).reshape(-1,self.vocab_size)