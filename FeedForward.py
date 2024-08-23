from torch import nn



class FeedForwardLayer(nn.Module):
    """Feedforward layer, that is placed after the multi-head
        attention layer"""
    def __init__(self,inter,output):
        super().__init__()
        self.output = output
        self.intermediate = inter
        self.firstfeedforward = nn.Linear(self.output,self.intermediate)
        self.secondfeedfoward = nn.Linear(self.intermediate,self.output)
        self.relu = nn.ReLU()

    def forward(self,x):
        """Forward pass of feedforward network."""
        x  = self.firstfeedforward(x)
        x  = self.relu(x)
        x  = self.secondfeedfoward(x)
        return x
