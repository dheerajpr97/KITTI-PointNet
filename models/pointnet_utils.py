import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STN3d(nn.Module):
    """
    Spatial Transformer Network for 3D data.
    It applies a spatial transformation to the input data (3D point cloud).
    """
    def __init__(self, channel):
        """
        Initialize the STN3d module.
        
        :param channel: Number of channels in the input data.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Forward pass of the STN3d module.
        
        :param x: Input tensor.
        :return: Transformed tensor.
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling over a (N, C) tensor
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity matrix initialization
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """
    Spatial Transformer Network for k-dimensional data.
    Applies a spatial transformation to the input data.
    """
    def __init__(self, k=64):
        """
        Initialize the STNkd module.

        :param k: Size of each input sample.
        """
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        """
        Forward pass of the STNkd module.

        :param x: Input tensor.
        :return: Transformed tensor.
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize with identity matrix
        iden = Variable(torch.eye(self.k).flatten().type_as(x)).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """
    Encoder network in PointNet architecture.
    Encodes input point cloud data for classification or segmentation tasks.
    """
    def __init__(self, num_classes=15, global_feat=True, feature_transform=False, channel=3):
        """
        Initialize the PointNetEncoder module.

        :param num_classes: Number of classes for classification.
        :param global_feat: Boolean indicating whether to use global features.
        :param feature_transform: Boolean indicating whether to use feature transform.
        :param channel: Number of channels in the input data.
        """
        super(PointNetEncoder, self).__init__()

        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """
        Forward pass of the PointNetEncoder module.

        :param x: Input tensor of point cloud data.
        :return: Encoded features, transformation matrices.
        """
        #print(f'Input shape: {x.shape}')
        B, D, N = x.size()
        #print(f'B: {B}, D: {D}, N: {N}')
        trans = self.stn(x)
        x = x.transpose(2, 1)

        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]

        x = torch.bmm(x, trans)

        if D > 3:
            x = torch.cat([x, feature], dim=2)

        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        #print(f'x.shape after conv: {x.shape}')

        if self.global_feat:
            return x, trans, trans_feat
        else:
            #print(f'x.shape before view: {x.shape}')
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            #print(f'x.shape after view: {x.shape}, pointfeat.shape: {pointfeat.shape}')
            return torch.cat([x, pointfeat], 1), trans, trans_feat

