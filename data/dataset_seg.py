import torch
from torch.utils.data import Dataset
import numpy as np
from utils.utils import normalize_point_cloud


def remap_labels(labels, label_mapping):
    remapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
    return remapped_labels

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
