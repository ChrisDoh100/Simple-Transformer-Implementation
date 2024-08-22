import torch
from torch.utils.data import Dataset




class TranslateDataset(Dataset):
    def __init__(self,firstlang,secondlang):
        self.firstlang = firstlang
        self.secondlang = secondlang
    
    def __len__(self):
        return len(self.firstlang)
    
    def __getitem__(self,idx):
        return self.firstlang[idx],self.secondlang[idx]