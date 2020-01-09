from torch import nn
from torch.nn import functional as F

class CnnModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # input 1 * 32 * 32

        self.conv_1_1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 64 * 30 * 30

        # self.conv_1_2 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        # ) # 64 * 28 * 28

        self.pooling_1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        ) # 64 * 14 * 14

        self.conv_2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 128 * 12 * 12

        # self.conv_2_2 = nn.Conv2d(
        #     in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        # ) # 128 * 12 * 12

        self.pooling_2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        ) # 128 * 5 * 5

        # self.fc_1 = nn.Linear(3200, 3200)

        self.fc_2 = nn.Linear(4608, 2)

    def forward(self, input_image):
        size_batch_in = input_image.size(0)

        out = self.conv_1_1(input_image)
        # out = self.conv_1_2(out)
        # out = F.relu(out)
        out = self.pooling_1(out)

        out = self.conv_2_1(out)
        # out = self.conv_2_2(out)
        out = F.relu(out)
        out = self.pooling_2(out)

        out = out.view(size_batch_in, -1)
        # out = self.fc_1(out)
        out = F.relu(out)
        out = F.dropout(out)
        out = self.fc_2(out)

        return out