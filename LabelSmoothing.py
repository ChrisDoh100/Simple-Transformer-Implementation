import torch
from torch import nn

class LabelSmoothingDistribution(nn.Module):
    """
        Instead of using cross-entropy where we have the target word set to 1 and the rest 0, we set
        the target embedding/word to a value like 0.9 and the distribute the remaining 0.1 amongst the 
        rest of the words, this makes the model more unsure and acts as a decent regularisation technique.
    """

    def __init__(self, smoothing_value, pad_token_id, trg_vocab_size, device):
        assert 0.0 <= smoothing_value <= 1.0

        super(LabelSmoothingDistribution, self).__init__()

        self.confidence_value = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value

        self.pad_token_id = pad_token_id
        self.trg_vocab_size = trg_vocab_size
        self.device = device

    def forward(self, trg_token_ids_batch):

        batch_size = trg_token_ids_batch.shape[0]
        smooth_target_distributions = torch.zeros((batch_size, self.trg_vocab_size), device='cuda')
        smooth_target_distributions.fill_(self.smoothing_value / (self.trg_vocab_size - 2))
        smooth_target_distributions.scatter_(1, trg_token_ids_batch, self.confidence_value)
        smooth_target_distributions[:, self.pad_token_id] = 0.
        smooth_target_distributions.masked_fill_(trg_token_ids_batch == self.pad_token_id, 0.)

        return smooth_target_distributions