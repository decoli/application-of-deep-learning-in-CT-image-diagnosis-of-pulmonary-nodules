import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

class ExtractingSemanticsModel(nn.Module):
    def __init__(self):
        super(ExtractingSemanticsModel, self).__init__()

        # CNN
        ## kernel = 3*3
        self.conv_3_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv_3_2 = nn.Conv2d(32, 32, 3, padding=1)
        #
        self.conv_3_3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_3_4 = nn.Conv2d(64, 64, 3, padding=1)
        #
        self.conv_3_5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_3_6 = nn.Conv2d(128, 128, 3, padding=1)
        #
        self.fc_3_1 = nn.Linear(2048, 512)

        ## kernel = 5*5
        self.conv_5_1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv_5_2 = nn.Conv2d(32, 32, 5, padding=2)
        #
        self.conv_5_3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv_5_4 = nn.Conv2d(64, 64, 5, padding=2)
        #
        self.conv_5_5 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv_5_6 = nn.Conv2d(128, 128, 5, padding=2)
        #
        self.fc_5_1 = nn.Linear(2048, 512)

        ## kernel = 7*7
        self.conv_7_1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conv_7_2 = nn.Conv2d(32, 32, 7, padding=3)
        #
        self.conv_7_3 = nn.Conv2d(32, 64, 7, padding=3)
        self.conv_7_4 = nn.Conv2d(64, 64, 7, padding=3)
        #
        self.conv_7_5 = nn.Conv2d(64, 128, 7, padding=3)
        self.conv_7_6 = nn.Conv2d(128, 128, 7, padding=3)
        #
        self.fc_7_1 = nn.Linear(2048, 512)

        # all
        self.fc_all_1 = nn.Linear(1536, 1536)
        self.fc_all_2 = nn.Linear(1536, 256)
        self.fc_all_3 = nn.Linear(256, 40)

    def forward(self, x_1, x_2, x_3):
        size_in = x_1.size(0)

        out_3 = self.conv_3_1(x_1)
        out_3 = self.conv_3_2(out_3)
        out_3 = F.max_pool2d(out_3, 2, 2)
        out_3 = F.relu(out_3)

        #
        out_3 = self.conv_3_3(out_3)
        out_3 = self.conv_3_4(out_3)
        out_3 = F.max_pool2d(out_3, 2, 2)
        out_3 = F.relu(out_3)

        #
        out_3 = self.conv_3_5(out_3)
        out_3 = self.conv_3_6(out_3)
        out_3 = F.max_pool2d(out_3, 2, 2)
        out_3 = F.relu(out_3)

        #
        out_3 = out_3.view(size_in, -1)
        out_3 = F.dropout(out_3)
        out_3 = self.fc_3_1(out_3)

        ## kernel=5*5
        out_5 = self.conv_5_1(x_2)
        out_5 = self.conv_5_2(out_5)
        out_5 = F.max_pool2d(out_5, 2, 2)
        out_5 = F.relu(out_5)

        #
        out_5 = self.conv_5_3(out_5)
        out_5 = self.conv_5_4(out_5)
        out_5 = F.max_pool2d(out_5, 2, 2)
        out_5 = F.relu(out_5)

        #
        out_5 = self.conv_5_5(out_5)
        out_5 = self.conv_5_6(out_5)
        out_5 = F.max_pool2d(out_5, 2, 2)
        out_5 = F.relu(out_5)

        #
        out_5 = out_5.view(size_in, -1)
        out_5 = F.dropout(out_5)
        out_5 = self.fc_5_1(out_5)
        
        ## kernel=7*7
        out_7 = self.conv_7_1(x_3)
        out_7 = self.conv_7_2(out_7)
        out_7 = F.max_pool2d(out_7, 2, 2)
        out_7 = F.relu(out_7)

        #
        out_7 = self.conv_7_3(out_7)
        out_7 = self.conv_7_4(out_7)
        out_7 = F.max_pool2d(out_7, 2, 2)
        out_7 = F.relu(out_7)

        #
        out_7 = self.conv_7_5(out_7)
        out_7 = self.conv_7_6(out_7)
        out_7 = F.max_pool2d(out_7, 2, 2)
        out_7 = F.relu(out_7)

        #
        out_7 = out_7.view(size_in, -1)
        out_7 = F.dropout(out_7)
        out_7 = self.fc_7_1(out_7)

        # all
        out_all = torch.cat([out_3, out_5, out_7], dim=1)
        out_all = self.fc_all_1(out_all)
        out_all = F.dropout(out_all)
        out_all = self.fc_all_2(out_all)
        out_all = F.relu(out_all)
        out_all = self.fc_all_3(out_all)

        return out_all

model = ExtractingSemanticsModel()
model.load_state_dict(torch.load('model_extracting_semantics.pt'))
print('dd')
print(model)