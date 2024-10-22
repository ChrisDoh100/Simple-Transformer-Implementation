from torch import nn



class FeedForwardLayer(nn.Module):
    
    def __init__(self, higher_dim,model_dimension):
        super().__init__()
        self.linear1 = nn.Linear(model_dimension, higher_dim)
        self.linear2 = nn.Linear(higher_dim, model_dimension)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.relu(self.linear1(representations_batch)))
