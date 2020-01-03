import torch.nn as nn
import torch

class AutoEncoderMalignant():
    def __init__(self):
        super(AutoEncoderMalignant, self).__init__()

        self.conv_down_01 = nn.Conv2d(1, 6, 3) # 30 * 30 * 6
        self.conv_down_02 = nn.Conv2d(6, 6, 3) # 28 * 28 * 6
        self.conv_down_03 = nn.Conv2d(6, 12, 3) # 26 * 26 * 12
        self.conv_down_04 = nn.Conv2d(12, 12, 3) # 24 * 24 * 12

        self.conv_up_01 = nn.Conv2d()
