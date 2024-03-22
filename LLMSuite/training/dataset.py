import pandas as pd 

class HF_Dataset():
    def __init__(self, data, split_ratio=0.05, seed=42):  
        self.data = data     
        self.split = self.data.train_test_split(test_size=split_ratio, seed=seed)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def train_test_split(self):
        return self.split['train'], self.split['test']
