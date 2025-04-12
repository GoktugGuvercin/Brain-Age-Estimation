
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BrainAgeCNN(nn.Module):
    """
    The BrainAgeCNN predicts the age given a brain MR-image.
    """
    def __init__(self, feats: int = 16, adap_pool: int = 8) -> None:
        super(BrainAgeCNN, self).__init__()

        # feature extractor
        self.conv1 = nn.Conv3d(1, feats, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm3d(feats)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.drop1 = nn.Dropout3d(p=0.4)

        self.conv2 = nn.Conv3d(feats, feats * 2, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm3d(feats * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.drop2 = nn.Dropout3d(p=0.4)

        self.conv3 = nn.Conv3d(feats * 2, feats * 4, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm3d(feats * 4)
        self.ada_pool = nn.AdaptiveAvgPool3d(adap_pool)  # (B, 64, 8, 8, 8)

        # predictor
        num_feats = (4 * feats) * (adap_pool ** 3)  # 32768 for 64,8,8,8
        self.fc1 = nn.Linear(num_feats, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.drop3 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        Args:
            imgs: Batch of input images. Shape (N, 1, H, W, D)
        """
        x = F.relu(self.bn1(self.conv1(imgs)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.ada_pool(x).flatten(start_dim=1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop3(x)

        out = self.fc2(x)
        return out
