import copy
import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset_seg import PointCloudDataset
from models.pointnet_seg import PointNetSeg, pointnet_loss
from training.train_utils import EarlyStopping, calculate_accuracy_seg, load_data, calculate_class_weights
from utils.utils import remap_labels

def train_one_epoch(model, dataloader, optimizer, device, class_weights, reg_weight, label_mapping):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, labels in dataloader:
        labels = labels.view(-1)
        labels = remap_labels(labels, label_mapping)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs, trans = model(data)
        loss = pointnet_loss(outputs, labels, trans, class_weights, reg_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_correct, batch_total = calculate_accuracy_seg(outputs, labels)
        correct += batch_correct
        total += batch_total
    
    return total_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, class_weights, device, reg_weight, label_mapping):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            labels = labels.view(-1)
            labels = remap_labels(labels, label_mapping)
            data, labels = data.to(device), labels.to(device)

            outputs, trans = model(data)
            loss = pointnet_loss(outputs, labels, trans, class_weights, reg_weight)

            total_loss += loss.item()
            batch_correct, batch_total = calculate_accuracy_seg(outputs, labels)
            correct += batch_correct
            total += batch_total
    
    return total_loss / len(dataloader), 100 * correct / total

def train(model, train_dataloader, val_dataloader, optimizer, device, class_weights_dict, epochs=10, 
          reg_weight=0.001, scheduler=None, early_stopping_patience=5, label_mapping=None):
    
    class_weights = torch.tensor(list(class_weights_dict.values()), dtype=torch.float32).to(device)
    
    print("Training started...")
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, optimizer, device, class_weights, reg_weight, 
                                                     label_mapping)
        val_loss, val_accuracy = validate(model, val_dataloader, class_weights,device, reg_weight, label_mapping)
        
        if scheduler:
            scheduler.step(val_loss)

        early_stopping.step(train_loss, model)
        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_model_state)
            break

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}], Duration: {epoch_duration:.2f} s, Learning Rate: {optimizer.param_groups[0]['lr']:.5f}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    print("Training finished.")
    # Load the best model
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    
    return model


def main(args):
    
    # Hyperparameters 
    NUM_POINTS = args.num_points
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = 15
    TASK = args.task


    # Load data
    train_points, train_labels = load_data(args.data_path + '/processed/train', TASK, NUM_POINTS, 'train')   
    val_points, val_labels = load_data(args.data_path + '/processed/val', TASK, NUM_POINTS, 'val')
    class_weights = calculate_class_weights(train_labels.flatten())

    TRAIN_DATASET = PointCloudDataset(train_points, train_labels)
    VAL_DATASET = PointCloudDataset(val_points, val_labels)
    
    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load the labels and create a label mapping
    filtered_labels = pd.read_csv(args.data_path + '/filtered_labels.csv')
    labels_unique = np.unique(filtered_labels['id'].values)
    label_mapping = {labels_unique: new_label for new_label, labels_unique in enumerate(labels_unique)} 

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = PointNetSeg(num_class=NUM_CLASSES).to(device)

    # Define the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.step_size) #step_size=args.step_size, gamma=args.gamma)

    # Train the model
    trained_model = train(model, train_dataloader, val_dataloader, optimizer, device=device,
                          epochs=args.epochs, reg_weight=args.reg_weight, scheduler=scheduler,
                          early_stopping_patience=args.early_stopping_patience, label_mapping=label_mapping, class_weights_dict=class_weights)
    
    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.save_model_path = args.save_model_path + args.task + '_' + timestamp + '.pt'

    # Save the trained model
    torch.save(trained_model.state_dict(), args.save_model_path)
    print("Model saved to ", args.save_model_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a PointNetSeg model.")
    parser.add_argument('--data_path', type=str, default='data', required=True, help='Path to training data')
    parser.add_argument('--task', type=str, required=True, choices=['cls', 'seg'], help='Specify "cls or "seg" respectively for classification or segmentation task')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points to downsample to')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=150, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for LR scheduler')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='Regularization weight')
    parser.add_argument('--early_stopping_patience', type=int, default=250, help='Early stopping patience')
    parser.add_argument('--save_model_path', type=str, default='models/trained_models/trained_model_', help='Path to save the trained model')

    args = parser.parse_args()
    main(args)

# Example usage: python -m training.train_seg --data_path data --task seg --num_points 1024 --batch_size 64 --learning_rate 0.01 --step_size 250 --gamma 0.5 --epochs 1250 --reg_weight 0.001 --early_stopping_patience 500