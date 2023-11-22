import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.dropout = dropout
        self.fc = self._make_fc_layers()

    def _make_fc_layers(self):
        """
        创建全连接层的私有辅助函数。

        Returns:
            nn.Sequential: 包含全连接层的序列模型。
        """
        layers = []
        in_features = self.in_features
        hidden_features = self.hidden_features

        for hidden_feature in hidden_features:
            layers.append(nn.Linear(in_features, hidden_feature))
            layers.append(nn.LayerNorm(hidden_feature, eps=1e-18))
            layers.append(nn.ReLU())  # 激活函数
            layers.append(nn.Dropout(p=self.dropout))
            in_features = hidden_feature

        layers.append(nn.Linear(in_features, self.out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
