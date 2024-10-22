
from torch.utils.data import Dataset
import torch




class TranslateDataset(Dataset):
    """Converts Source sentences, target sentences(split into decoder inputs and outputs) into tensors."""
    
    def __init__(self,sourceinput,targetinput,groundtruth):
        self.sourceinput = [torch.tensor(i) for i in sourceinput]
        self.targetinput = [torch.tensor(i) for i in targetinput]
        self.groundtruth = [torch.tensor(i) for i in groundtruth]
        
    def __len__(self):
        return len(self.sourceinput)
    
    def __getitem__(self,idx):
        return self.sourceinput[idx],self.targetinput[idx],self.groundtruth[idx]