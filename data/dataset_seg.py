import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import normalize_point_cloud

class PointCloudDataset(Dataset):
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels
       
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point_cloud = normalize_point_cloud(self.points[idx])
        point_cloud = torch.tensor(point_cloud, dtype=torch.float).T
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return point_cloud, label
