# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, (1, 64), stride=2),
            nn.MaxPool1d((1, 8), stride=8),
            nn.Conv1d(1, 32, (1, 32), stride=2),
            nn.MaxPool1d((1, 8), stride=8),
            nn.Conv1d(1, 64, (1, 16), stride=2),
            nn.Conv1d(1, 128, (1, 8), stride=2),
        )
        self.fc = nn.Sequential(nn.Linear(128 * 8, 128), nn.Linear(128, 64))

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.softmax(x)

