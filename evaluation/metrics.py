# Custom metrics for model evaluation

import numpy as np

def calculate_miou(ground_truth, prediction, num_classes):
    miou = 0.0
    for cls in range(num_classes):
        intersection = np.logical_and(ground_truth == cls, prediction == cls)
        union = np.logical_or(ground_truth == cls, prediction == cls)
        
        if np.sum(union) == 0:
            # If there is no ground truth and no prediction for this class, skip it
            continue
        
        iou = np.sum(intersection) / np.sum(union)
        miou += iou

    miou /= num_classes
    return miou