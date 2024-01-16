import torch
import numpy as np
from collections import defaultdict
import random


class PointCloudProcessorCls:
    def __init__(self, num_points, trainid_to_keep, num_chunks_per_class=100):
        self.num_points = num_points
        self.trainid_to_keep = trainid_to_keep
        self.num_chunks_per_class = num_chunks_per_class

    def process_point_cloud(self, raw_point_cloud):
        filtered_pc = self.filter_point_cloud(raw_point_cloud, self.trainid_to_keep)
        chunked_data = self._generate_chunks(filtered_pc, self.num_points)
        stratified_data = self._stratify_chunks(chunked_data)
        
        for chunk, label in stratified_data:
            normalized_chunk = self._normalize_chunk(chunk)
            yield torch.tensor(normalized_chunk, dtype=torch.float32), label

    def filter_point_cloud(self, point_cloud, trainid_to_keep):
        mask = np.isin(point_cloud['semantic'], trainid_to_keep)
        filtered_pc = point_cloud[mask]
        pc_label = np.vstack([filtered_pc['x'], filtered_pc['y'], filtered_pc['z'], filtered_pc['semantic']]).T    
        return pc_label

    def _interpolate_points(self, point1, point2):
        return tuple((np.array(point1) + np.array(point2)) / 2)

    def _generate_chunks(self, point_cloud, num_points):
        grouped_points = defaultdict(list)
        for x, y, z, label in point_cloud:
            grouped_points[label].append((x, y, z))

        chunked_data = defaultdict(list)
        for label, points in grouped_points.items():
            for i in range(0, len(points), num_points):
                chunk = points[i:i+num_points]
                while len(chunk) < num_points:
                    if len(chunk) >= 2:
                        new_point = self._interpolate_points(random.choice(chunk), random.choice(chunk))
                        chunk.append(new_point)
                    elif len(chunk) == 1:
                        chunk.append(chunk[0])
                #yield chunk, label
                    chunked_data[label].append(chunk)

        return chunked_data            

    def _stratify_chunks(self, chunked_data):
        stratified_data = []
        for label, chunks in chunked_data.items():
            sampled_chunks = chunks if len(chunks) <= self.num_chunks_per_class else random.sample(chunks, self.num_chunks_per_class)
            stratified_data.extend([(chunk, label) for chunk in sampled_chunks])

        random.shuffle(stratified_data)
        return stratified_data

    def _normalize_chunk(self, chunk):
        chunk = np.array(chunk)
        centroid = np.mean(chunk, axis=0)
        centered_chunk = chunk - centroid
        max_distance = np.max(np.sqrt(np.sum(centered_chunk**2, axis=1)))
        # Add an epsilon to max_distance to avoid division by zero
        epsilon = 1e-6
        normalized_chunk = centered_chunk / (max_distance + epsilon)
        
        return normalized_chunk
    

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Convert data to Torch Tensors
        pointcloud = torch.from_numpy(self.data[idx]).T
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pointcloud, label


