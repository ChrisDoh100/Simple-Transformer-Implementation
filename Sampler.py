
import torch
import torch.utils
import torch.utils.data
from torch.utils.data.sampler import Sampler
import random

class tokenBatchSampler(Sampler):
    """Sampling batches from the dataset to ensure roughly similar batch sizes based on token amount."""

    def __init__(self,dataset,max_tokens=1500,shuffle=True):
        self.dataset=dataset
        self.max_tokens=max_tokens
        self.shuffle = shuffle
        self.batches = self.__createbatches__()

    def __createbatches__(self):
        """Created batches for the the dataloader, based on token size."""
        batches=[]
        current_tokens=0
        current_batch=[]
        sorted_indices = sorted(range(len(self.dataset)), key=lambda i: max(len(self.dataset[i][0]), len(self.dataset[i][1])))
        for y in sorted_indices:
            current_length=max(len(self.dataset[y][0]),len(self.dataset[y][1]))
            if (current_length+current_tokens)>self.max_tokens:
                if current_batch and current_tokens<=self.max_tokens:
                    batches.append(current_batch)
                    if current_length<=self.max_tokens:
                        current_tokens=current_length
                        current_batch=[y]
                    else:
                        current_tokens=0
                        current_batch=[]
            else:
                current_tokens+=current_length
                current_batch.append(y)

        if current_batch:
            batches.append(current_batch)
        if self.shuffle:
            for batch in batches:
                random.shuffle(batch)
            random.shuffle(batches)
        return batches

    def __iter__(self):
        #Very important to shuffle the batches before
        #returning an iterable, because if you don't you introduce 
        #a strong inductive bias as you've sorted the sequences by length.
        if self.shuffle:
            for batch in self.batches:
                random.shuffle(batch)
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)