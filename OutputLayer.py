from torch import nn

class DecoderOutputGenerator(nn.Module):
    """Final module for the transformer that generates our output."""
    def __init__(self,config):
        super().__init__()
        self.vocab_size = config['vocab_size']
        self.dmodel = config['model_dimension']
        self.lin = nn.Linear(self.dmodel,self.vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self,x):
        """Forward pass that generates the final output."""
        return self.logsoftmax(self.lin(x)).reshape(-1,self.vocab_size)