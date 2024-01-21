import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import normalize_point_cloud

class PointCloudDataset(Dataset):
    def __init__(self, points, labels):
        """
        Args:
            points (list of numpy arrays): List of point cloud data (each array: num_points x 3).
            labels (list of int): List of labels corresponding to each point cloud.
        """
        self.points = points
        self.labels = labels
        

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        
        # Normalize the point cloud
        point_cloud = normalize_point_cloud(self.points[idx])
        
        # Convert data to Torch Tensors
        point_cloud = torch.tensor(point_cloud, dtype=torch.float).T
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return point_cloud, label
