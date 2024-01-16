import os

import numpy as np
import pandas as pd
import torch

from utils.ply import read_ply


def normalize_point_cloud(points):
    """
    Normalize a point cloud to the origin and scale it to fit inside a unit sphere.

    Args:
        points (np.ndarray): The Nx3 array of points.
    Returns:
        np.ndarray: Normalized point cloud.
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points /= furthest_distance
    return points

def filter_point_cloud(point_cloud, trainid_to_keep):
    """
    Filter a point cloud based on a list of 'trainid' values.

    Args:
        point_cloud (dict): The point cloud data.
        trainid_to_keep (list): List of 'trainid' values to keep.

    Returns:
        tuple: Filtered point cloud coordinates and semantic labels.
    """
    mask = np.isin(point_cloud['semantic'], trainid_to_keep)
    return point_cloud[mask]

def load_ply_files_from_directory(directory):
    """
    Load all .ply files from the specified directory.

    Args:
        directory (str): The path to the directory containing .ply files.

    Returns:
        list of open3d.geometry.PointCloud: A list of PointCloud objects.
    """
    points = []
    point_clouds = []
    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            file_path = os.path.join(directory, filename)
            point_cloud = read_ply(file_path)
            #points.append(np.vstack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).T)
            point_clouds.append(point_cloud)
    return point_clouds

def load_point_clouds_from_paths(file_path, base_dir='data/'):
    """
    Load file paths from a text file and format them.

    Args:
        file_path (str): Path to the text file containing point cloud file paths.
        base_dir (str): Base directory to prepend to the file paths (default is 'data/').

    Returns:
        list: List of formatted file paths.
    """
    point_cloud_paths = []
    with open(file_path, 'r') as f:
        file_paths = f.readlines()
        for path in file_paths:
            file_path = path.strip()
            full_path = base_dir + file_path
            point_cloud_paths.append(full_path)
    
    return point_cloud_paths

def create_point_arrays(df):
    """
    Create arrays of (x, y, z, semantic) from a DataFrame.

    Args:
        df (DataFrame): DataFrame containing 'x', 'y', 'z', and 'semantic' columns.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each representing a point cloud.
    """
    point_arrays = []
    for _, row in df.iterrows():
        points = np.column_stack((row['x'], row['y'], row['z'], row['semantic']))
        point_arrays.append(points)
    return point_arrays

def approximate_fps(point_cloud, labels, num_samples=256):
    """
    Perform an approximate Farthest Point Sampling (FPS) on a point cloud.

    Args:
        point_cloud (np.ndarray): Numpy array representing the point cloud (N x D).
        labels (np.ndarray): Numpy array of labels corresponding to the point cloud.
        num_samples (int): Number of samples to select.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of sampled points and their labels.
    """
    num_points = point_cloud.shape[0]
    sampled_indices = [np.random.choice(num_points)]
    distances = np.full(num_points, np.inf)
    
    while len(sampled_indices) < num_samples:
        last_index = sampled_indices[-1]
        new_distances = np.linalg.norm(point_cloud - point_cloud[last_index], axis=1)
        distances = np.minimum(distances, new_distances)
        farthest_point = np.argmax(distances)
        sampled_indices.append(farthest_point)

    sampled_points = point_cloud[sampled_indices, :]
    sampled_labels = labels[sampled_indices]

    return sampled_points, sampled_labels


def load_point_clouds_in_chunks(file_paths, chunk_size=10):
    """
    Generator that loads point clouds in chunks from given file paths.

    This function is a generator, yielding point clouds loaded in batches. 
    Each batch is loaded by reading point cloud data from a set of file paths.

    Args:
        file_paths (List[str]): List of file paths to point cloud files.
        chunk_size (int): The number of files to load in each chunk. Defaults to 10.

    Yields:
        List: A list of point clouds loaded from the current chunk of file paths.
    """
    for i in range(0, len(file_paths), chunk_size):
        chunk = file_paths[i:i + chunk_size]
        point_clouds = [read_ply(file_path) for file_path in chunk]
        yield point_clouds

def create_point_arrays_from_filtered(filtered_points):
    """
    Create point arrays from filtered point cloud data.

    Args:
        filtered_points (dict): A dictionary containing 'x', 'y', 'z', and 'semantic' keys
                                with their corresponding values as arrays.

    Returns:
        List[np.ndarray]: A list containing a numpy array of the stacked point cloud data.
    """
    x = filtered_points['x']
    y = filtered_points['y']
    z = filtered_points['z']
    semantic = filtered_points['semantic']

    point_arrays = [np.column_stack((x, y, z, semantic))]
    return point_arrays


def downsample_point_cloud_chunks_seg(file_paths, trainid_to_keep, chunk_size=1, num_samples=256):
    """
    Process point cloud data in chunks.

    Args:
        file_paths (list): List of file paths to point cloud files.
        trainid_to_keep (list): List of 'trainid' values to keep.
        chunk_size (int): Number of files to process in each chunk.
        num_samples (int): Number of samples to downsample to.

    Returns:
        tuple: Lists of downsampled points and labels.
    """
    downsampled_points, downsampled_labels = [], []

    for point_cloud_chunk in load_point_clouds_in_chunks(file_paths=file_paths, chunk_size=chunk_size):
        for point_cloud in point_cloud_chunk:
            filtered_points = filter_point_cloud(point_cloud, trainid_to_keep)
            point_arrays = create_point_arrays_from_filtered(filtered_points)
            
            for point_array in point_arrays:
                sampled_points, sampled_labels = approximate_fps(point_array[:, :3], point_array[:, 3], num_samples=num_samples)
                downsampled_points.append(sampled_points)
                downsampled_labels.append(sampled_labels)

    sampled_points = np.asarray(downsampled_points)
    sampled_labels = np.asarray(downsampled_labels)
    
    return sampled_points, sampled_labels

def downsample_point_cloud_chunks_cls(file_paths, pc_processor, chunk_size=1):
    """
    Processes point clouds from file paths and collects chunks and labels.

    This function iterates over point cloud data in chunks, normalizes them, 
    and collects the chunks with their corresponding labels using a 
    PointCloudProcessor instance.

    Args:
    - file_paths (list of str): List of paths to point cloud files.
    - pc_processor (PointCloudProcessor): An instance of PointCloudProcessor to process point clouds.
    - chunk_size (int): The size of each chunk to process. Default is 1.

    Returns:
    - list: A list of processed and normalized point cloud chunks.
    - list: A list of corresponding labels for the point cloud chunks.
    """
    
    chunks = []
    labels = []

    for point_cloud_chunk in load_point_clouds_in_chunks(file_paths=file_paths, chunk_size=chunk_size):
        for normalized_chunk, label in pc_processor.process_point_cloud(np.asarray(point_cloud_chunk)):
            chunks.append(normalized_chunk)
            labels.append(label)

    return chunks, labels

def save_downsampled_data(downsampled_points, downsampled_labels, num_points, task='seg', data_type='train', directory='data/processed/'):
    """
    Save downsampled point cloud data and labels to numpy files.

    Args:
        downsampled_points (np.ndarray): Downsampled point cloud data.
        downsampled_labels (np.ndarray): Downsampled point cloud labels.
        num_points (int): Number of points in the downsampled data.
        directory (str, optional): Directory where files will be saved. Defaults to 'data'.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    points_file_path = os.path.join(directory, f'downsampled_points_{task}_{num_points}_{data_type}.npy')
    labels_file_path = os.path.join(directory, f'downsampled_labels_{task}_{num_points}_{data_type}.npy')

    np.save(points_file_path, downsampled_points)
    np.save(labels_file_path, downsampled_labels)

    print(f'Downsampled data saved as {points_file_path} and {labels_file_path}')


def remap_labels(labels, label_mapping):
    remapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
    return remapped_labels