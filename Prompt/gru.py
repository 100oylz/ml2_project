import torch
import torch.nn as nn


class gru(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super().__init__()
        self.net = nn.GRU(in_features, out_features, num_layers=num_layers)

    def forward(self, x):
        x, _ = self.net(x)
        return x.view(1, -1)
