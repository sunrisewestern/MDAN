from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels, domains,ids):
        self.data = data
        self.labels = labels
        self.domains = domains
        self.ids = ids

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domains[index]
        i = self.ids[index]
        return x, y, d, i

    def __len__(self):
        return len(self.data)
    
class UnsqueezedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y, d, i = self.dataset[index]
        return torch.unsqueeze(x, 0), y, d, i # Unsqueezing the data

    def __len__(self):
        return len(self.dataset)