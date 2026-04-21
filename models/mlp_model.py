import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.fc1 = nn.Linear(in_features = 21, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 128)
        self.fc3 = nn.Linear(in_features = 128, out_features = 64)
        self.fc4 = nn.Linear(in_features = 64, out_features = 32)
        self.fc5 = nn.Linear(in_features = 32, out_features = 1)

        self.dropout = nn.Dropout(p)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.fc1(x))))
        out = self.dropout(F.relu(self.bn2(self.fc2(out))))
        out = self.dropout(F.relu(self.bn3(self.fc3(out))))
        out = self.dropout(F.relu(self.bn4(self.fc4(out))))
        out = self.fc5(out)

        return out

        



