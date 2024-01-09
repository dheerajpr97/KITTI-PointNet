# PointNet model architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from models.pointnet_utils import PointNetEncoder


class PointNetSeg(nn.Module):
    def __init__(self, num_class):
        """
        Initializes a new instance of the PointNetSeg class.

        Args:
            num_class (int): The number of classes.

        Returns:
            None
        """
        super(PointNetSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = torch.nn.Conv1d(1088 , 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)  

    def forward(self, x):
        #print("Type of data:", type(x))
        #print("Shape of data:", x.shape if isinstance(x, torch.Tensor) else [d.shape for d in x])

        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat
    
def pointnet_loss(outputs, labels, transform, device, reg_weight=0.001):
    """
        Calculates the PointNet loss function.

        Args:
            outputs (torch.Tensor): The predicted outputs of the model. Shape [batch_size * num_points, num_classes].
            labels (torch.Tensor): The ground truth labels. Shape [batch_size * num_points].
            transform (torch.Tensor): The predicted transformation matrix. Shape [batch_size, num_points, num_points].
            reg_weight (float, optional): The weight for the regularization term. Defaults to 0.001.

        Returns:
            torch.Tensor: The total loss calculated as the sum of the classification loss and the regularization loss.
    """

    outputs = outputs.view(-1, 15)  # Reshape to [64*128, 15]    
    classify_loss = F.nll_loss(outputs, labels) 

    # Compute the transformation regularization term
     
    K = transform.size(1)
    I = torch.eye(K).to(device) #[None, :, :]
    transform = transform.to(device)
    mat_diff_loss = torch.mean(torch.norm(torch.bmm(transform, transform.transpose(2,1)) - I, dim=(1,2)))
    
    total_loss = classify_loss + reg_weight * mat_diff_loss
    return total_loss
