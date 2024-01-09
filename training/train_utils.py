import copy
import os

import numpy as np
import torch


def load_data(path, num_points):
    point_path = os.path.join(path, f'downsampled_points_{num_points}.npy')
    label_path = os.path.join(path, f'downsampled_labels_{num_points}.npy')
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

