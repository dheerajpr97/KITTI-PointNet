import argparse
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import calculate_miou
from data.dataset_seg import PointCloudDataset
from models.pointnet_seg import PointNetSeg
from utils.ply import read_ply
from utils.utils import remap_labels, convert_color_string_to_tuple
from utils.vis import visualize_point_cloud, prepare_data_vis


def load_model_for_inference(model_path, num_classes):
    # Initialize the model
    model = PointNetSeg(num_class=num_classes)
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_point_cloud(point_cloud_path, filtered_labels_path):
    """
    Preprocess the point cloud by filtering according to provided labels and converting to tensors.
    
    Parameters:
    - point_cloud_path: str, path to the point cloud file (.ply)
    - filtered_labels_path: str, path to the CSV file containing filtered labels
    
    Returns:
    - point_cloud: torch.Tensor, the preprocessed point cloud
    - label: torch.Tensor, the corresponding labels
    """
    # Read the point cloud data
    test_point_cloud = read_ply(point_cloud_path)
    
    # Load filtered labels
    filtered_labels = pd.read_csv(filtered_labels_path)
    trainid_to_keep = filtered_labels['id'].values
    labels_unique = np.unique(filtered_labels['id'])
    label_mapping = {label: i for i, label in enumerate(labels_unique)}

    # Filter the point cloud
    mask = np.isin(test_point_cloud['semantic'], trainid_to_keep)
    filtered_point_cloud = test_point_cloud[mask]
    
    # Stack and transpose to get the correct shape
    points = np.vstack((filtered_point_cloud['x'], filtered_point_cloud['y'], filtered_point_cloud['z'])).T
    labels = filtered_point_cloud['semantic']
    
    # Convert to PyTorch tensors
    point_cloud = torch.tensor(points, dtype=torch.float)
    label = torch.tensor(labels, dtype=torch.long)
    
    return point_cloud, label, label_mapping, filtered_labels

def chunk_point_cloud_and_labels(point_cloud, labels, chunk_size=256):
    # Calculate the number of chunks needed
    num_chunks = len(point_cloud) // chunk_size
    
    # If the point cloud size isn't a multiple of chunk_size, add one more chunk for the remainder
    if len(point_cloud) % chunk_size != 0:
        num_chunks += 1

    # Create chunks for both point cloud and labels
    point_cloud_chunks = []
    label_chunks = []

    for i in range(num_chunks):
        # Calculate start and end indices for the chunk
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Chunk the point cloud
        pc_chunk = point_cloud[start_idx:end_idx]

        # Chunk the labels
        label_chunk = labels[start_idx:end_idx]

        # Ensure each chunk has the same size by padding if necessary
        if len(pc_chunk) < chunk_size:
            pc_padding_size = chunk_size - len(pc_chunk)
            pc_padding = torch.zeros((pc_padding_size, point_cloud.size(1)), dtype=point_cloud.dtype)
            pc_chunk = torch.cat([pc_chunk, pc_padding], dim=0)

            label_padding_size = chunk_size - len(label_chunk)
            label_padding = torch.zeros((label_padding_size), dtype=labels.dtype)
            label_chunk = torch.cat([label_chunk, label_padding], dim=0)

        point_cloud_chunks.append(pc_chunk)
        label_chunks.append(label_chunk)

    return np.array(point_cloud_chunks[:-1]), np.array(label_chunks[:-1])

def predict_and_evaluate_on_chunks(model, dataloader, device, num_classes, label_mapping):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():  # Turn off gradients for prediction
        for pc_chunk, label_chunk in dataloader:
            
            pc_chunk = pc_chunk.to(device)
            label_chunk = label_chunk.view(-1)
            label_chunk = remap_labels(label_chunk, label_mapping)
                        
            outputs, _ = model(pc_chunk)

            _, predicted = torch.max(outputs, 2)
            predicted = predicted.view(-1).cpu().numpy()

            all_predictions.extend(predicted)
            all_labels.extend(label_chunk)
            
            # Update confusion matrix
            conf_matrix += confusion_matrix(all_labels, all_predictions, labels=np.arange(num_classes))

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    #Calculate IoU for each class
    miou_score = calculate_miou(np.array(all_labels), np.array(all_predictions), num_classes)
    print(f"mIoU Score: {miou_score * 100:.2f}%") 

   
    return np.array(all_labels), np.array(all_predictions)

def map_semantics_to_colors(test_dataframe, filtered_labels, convert_color_string_to_tuple):
    """
    Maps semantic IDs to colors for predicted and true semantics, and applies a color conversion.
    
    Parameters:
    - test_dataframe: pandas.DataFrame, the dataframe with predicted and true semantics.
    - filtered_labels: pandas.DataFrame, the dataframe containing mapping from semantic ID to color.
    - convert_color_string_to_tuple: function, a function to convert color strings into color tuples.
    
    Returns:
    - test_dataframe: pandas.DataFrame, the dataframe with new columns for predicted and true colors.
    """
    # Create a mapping from semantic ID to color
    id_to_color = filtered_labels.set_index('mapped_id')['color']
    
    # Map predicted semantics to colors and apply color conversion
    test_dataframe['Predicted color'] = test_dataframe['Predicted semantic'].map(id_to_color)
    test_dataframe['Predicted color'] = test_dataframe['Predicted color'].apply(convert_color_string_to_tuple)
    
    # Map true semantics to colors and apply color conversion
    test_dataframe['True color'] = test_dataframe['True semantic'].map(id_to_color)
    test_dataframe['True color'] = test_dataframe['True color'].apply(convert_color_string_to_tuple)
    
    return test_dataframe



def run_inference(model, point_cloud_path, filtered_labels_path, num_points, num_classes, device='cpu'):
    time_start = time.time()    
    # Process the point cloud
    point_cloud, labels, label_mapping, filtered_labels = preprocess_point_cloud(point_cloud_path, filtered_labels_path)
    chunk_pc, chunk_label = chunk_point_cloud_and_labels(point_cloud, labels, chunk_size=num_points)
    original_pc = np.copy(chunk_pc.reshape(-1, 3))
    print('Point cloud loaded and preprocessing completed...')    

    # Create the dataset
    dataset = PointCloudDataset(chunk_pc, chunk_label)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Run inference
    model = model.to(device)
    all_labels, all_predictions = predict_and_evaluate_on_chunks(model, dataloader, device, num_classes, label_mapping)
    print('Inference completed...')

    # Postprocess and visualize the results
    
    test_dataframe = pd.DataFrame({'x': original_pc[:, 0], 'y': original_pc[:, 1], 'z': original_pc[:, 2],
                                   'True semantic': all_labels, 'Predicted semantic': all_predictions})

    # Map semantics to colors
    test_dataframe = map_semantics_to_colors(test_dataframe, filtered_labels, convert_color_string_to_tuple)

    # Visualize the results
    print('Visualizing the results...')
    pc, true_color = prepare_data_vis(test_dataframe, color_type='True color')
    visualize_point_cloud(pc, true_color, window_name='True Point Cloud', save_path='true_point_cloud.png')

    pc, pred_color = prepare_data_vis(test_dataframe, color_type='Predicted color')
    visualize_point_cloud(pc, pred_color, window_name='Predicted Point Cloud', save_path='predicted_point_cloud.png')

    print(f"Time taken for inference: {time.time() - time_start:.2f} seconds")
    


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on a point cloud using a trained PointNet model.')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file.')
    parser.add_argument('--point_cloud_path', type=str, help='Path to the point cloud data file.')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes.')
    parser.add_argument('--filtered_labels_path', type=str, default='data/filtered_labels.csv', help='Path to the CSV file containing filtered labels.')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points in the point cloud.')
    args = parser.parse_args()

    # Load the model
    model = load_model_for_inference(args.model_path, args.num_classes)

    # Run inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_inference(model, args.point_cloud_path, args.filtered_labels_path, args.num_points, args.num_classes, device)

    
if __name__ == '__main__':
    main()

# Example usage:
# python -m evaluation.evaluate_seg --model_path models/trained_models/trained_model_seg_20240119-164702.pt --point_cloud_path data/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000002_0000000385.ply --num_classes 15  --filtered_labels_path data/filtered_labels.csv --num_points 1024