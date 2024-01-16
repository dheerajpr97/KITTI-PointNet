import argparse
import numpy as np
import pandas as pd
from utils.ply import read_ply
from utils.utils import *
import time
from data.dataset_cls import PointCloudProcessorCls

def main(args):
    # Add time taken to run the script
    start_time = time.time()
    
    # Load filtered labels
    filtered_labels = pd.read_csv(args.filtered_labels_path)
    trainid_to_keep = filtered_labels['id']

    # Determine which file paths to use based on data_type
    if args.data_type == 'train':
        file_paths = load_point_clouds_from_paths(args.train_file_path)
        print(f'Length of train files: {len(file_paths)}')
        print(f'Train files paths loaded...')

    elif args.data_type == 'val':
        file_paths = load_point_clouds_from_paths(args.val_file_path)
        print(f'Length of val files: {len(file_paths)}')
        print(f'Val files paths loaded...')

    else:
        raise ValueError("Invalid data_type specified. Choose 'train' or 'val'.")
    
    # Process point clouds
    print(f'Processing point clouds started...')
    pc_processor = PointCloudProcessorCls(num_points=args.num_points, trainid_to_keep=trainid_to_keep, 
                                       num_chunks_per_class=10) 
    sampled_points, sampled_labels = downsample_point_cloud_chunks_cls(file_paths, 
                                                                       pc_processor)
    print(f'Point clouds processed...')

    # Save the processed data
    SAVE_DIR = os.path.join('data/processed', args.data_type)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    save_downsampled_data(sampled_points, sampled_labels, args.num_points, args.task, args.data_type, directory=SAVE_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process point cloud data.')
    parser.add_argument('--data_type', type=str, required=True, choices=['train', 'val'], help='Specify "train" or "val" for the type of data to process')
    parser.add_argument('--task', type=str, required=True, choices=['cls', 'seg'], help='Specify "cls or "seg" respectively for classification or segmentation task')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points to downsample to')
    parser.add_argument('--train_file_path', type=str, default='data/filepaths_train.txt', help='Path to train file list')
    parser.add_argument('--val_file_path', type=str, default='data/filepaths_val.txt', help='Path to validation file list')
    parser.add_argument('--filtered_labels_path', type=str, default='data/filtered_labels.csv', required=True, help='Path to filtered labels CSV file')
    args = parser.parse_args()
    main(args)

# Example usage command:
# python -m data.preprocess_scripts.data_cls --task cls --data_type val --num_points 512 --train_file_path data/filepaths_train.txt --val_file_path data/filepaths_val.txt --filtered_labels_path data/filtered_labels.csv