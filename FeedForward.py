from torch import nn



class FeedForwardLayer(nn.Module):
    """Feedforward layer of the transformer, allows connections between heads to be produced."""
    def __init__(self, config):
        super().__init__()
        self.dmodel = config['model_dimension']
        self.ffscale = config['feedforward_scale']
        self.linear1 = nn.Linear(self.dmodel, self.ffscale*self.dmodel)
        self.linear2 = nn.Linear(self.ffscale*self.dmodel, self.dmodel)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.relu(self.linear1(representations_batch)))
