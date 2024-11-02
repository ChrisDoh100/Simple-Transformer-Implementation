from torch import nn



class FeedForwardLayer(nn.Module):
    """Feedforward layer of the transformer, allows connections between heads to be realised."""
    def __init__(self, config):
        super().__init__()
        self.dmodel = config['model_dimension']
        self.ffscale = config['feedforward_scale']
        self.linear1 = nn.Linear(self.dmodel, self.ffscale*self.dmodel) # Increasing the dimensionality by some scale factor*model dimensionality
        self.linear2 = nn.Linear(self.ffscale*self.dmodel, self.dmodel)
        self.relu = nn.ReLU()

    def forward(self, X):
        #FeedForward Block 1
        # Relu(Linear(X))
        linear_input = self.linear1(X) #=Linear(X)
        relu_input = self.relu(linear_input)# = Relu(Linear(X))
        #FeedForward Block 2
        # Linear(Relu(Linear(X)))
        output = self.linear2(relu_input)#=Linear(Relu(Linear(X))) 
        return output
