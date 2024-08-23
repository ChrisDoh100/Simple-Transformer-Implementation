
from torch.utils.data import Dataset




class TranslateDataset(Dataset):
    """Converts two different set sof sentences into a huggingface face dataset."""
    def __init__(self,firstlang,secondlang):
        self.firstlang = firstlang
        self.secondlang = secondlang
    
    def __len__(self):
        return len(self.firstlang)
    
    def __getitem__(self,idx):
        return self.firstlang[idx],self.secondlang[idx]