import copy
import os

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def load_data(path, task, num_points, data_type):
    point_path = os.path.join(path, f'downsampled_points_{task}_{num_points}_{data_type}.npy')
    label_path = os.path.join(path, f'downsampled_labels_{task}_{num_points}_{data_type}.npy')
    points = np.load(point_path)
    labels = np.load(label_path)
    return points, labels


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                self.early_stop = True

def calculate_accuracy_seg(outputs, labels):
    _, predicted = torch.max(outputs, 2)
    predicted = predicted.view(-1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total

def calculate_accuracy_cls(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.view(-1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total



def calculate_class_weights(labels):
    # Calculate class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    # Calculate class weights
    classes = np.unique(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    return class_weights_dict