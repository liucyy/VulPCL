import pandas as pd
import torch

class DatasetIterdtor:
    '''生成可迭代数据集'''
    def __init__(self, batches, batch_size, device):
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.batch_size = batch_size
        self.device = device
        self.residue = False  # 记录batch数量是否为正数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def _to_tensor(self, data):
        s = torch.tensor([item[1] for item in data]).to(self.device)
        x1 = torch.LongTensor([item[2] for item in data]).to(self.device)
        x2 = torch.FloatTensor([item[3] for item in data]).to(self.device)
        label = torch.LongTensor([int(item[4]) for item in data]).to(self.device)
        return s, x1, x2, label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches