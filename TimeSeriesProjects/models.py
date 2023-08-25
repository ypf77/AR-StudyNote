import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=24 * 51, out_features=500, bias=True),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=500, out_features=500, bias=True),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=500, out_features=500, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=500, out_features=6, bias=True)
        )

    def forward(self, input):
        return self.net(input)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),

            nn.Linear(in_features=128, out_features=6)
        )

    def forward(self, input):
        return self.net(input)


# -------------------------------------模型测试---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MLP().to(device)
model = FCN().to(device)
# 输出每一层维度
summary(model, (24, 51))
