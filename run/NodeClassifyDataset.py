import torch
from torch_geometric.data import Dataset, Data
import numpy as np 
import os


class NodeClassifyDataset(Dataset):
    def __init__(self, root, name, filename, test, x, y, edge_label_index):
        
        self.new_dir = os.path.join("datasets", name, "processed")# "datasets\\" + name + "\\processed"   # new processed dir

        data = Data()
        tuple = (data, None)

        data.x = torch.tensor(x)
        data.y = torch.tensor(y)
        data.edge_index = torch.tensor(edge_label_index)
        torch.save(tuple, os.path.join(self.new_dir, 'all_data.pt'))


    def process(self):
        pass
