import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

class ManifoldEncoderS(nn.Module):
    def __init__(self):
        super(ManifoldEncoderS, self).__init__()
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 32)
        self.fc3 = nn.Linear(32, 48)
        self.fc4 = nn.Linear(48, 64)

    def forward(self, input):
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x


class ManifoldEncoderX(nn.Module):
    def __init__(self):
        super(ManifoldEncoderX, self).__init__()
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 32)
        self.fc3 = nn.Linear(32, 48)
        self.fc4 = nn.Linear(48, 64)

    def forward(self, input):
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x


class ManifoldEncoderY(nn.Module):
    def __init__(self):
        super(ManifoldEncoderY, self).__init__()
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 32)
        self.fc3 = nn.Linear(32, 48)
        self.fc4 = nn.Linear(48, 64)

    def forward(self, input):
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x



