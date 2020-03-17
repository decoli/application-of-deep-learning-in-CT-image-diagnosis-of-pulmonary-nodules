'''
融合先验知识（如医师标注的各结节征象），进行良恶性分类的模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

class PriorKnowledgeNet(nn.modules):
    def __init__(self):
        super(PriorKnowledgeNet, self).__init__()

        ## kernel = 3*3
        self.conv_3_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv_3_2 = nn.Conv2d(32, 32, 3, padding=1)
        #
        self.conv_3_3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_3_4 = nn.Conv2d(64, 64, 3, padding=1)
        #
        self.conv_3_5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_3_6 = nn.Conv2d(128, 128, 3, padding=1)

        ## kernel = 5*5
        self.conv_5_1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv_5_2 = nn.Conv2d(32, 32, 5, padding=2)
        #
        self.conv_5_3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv_5_4 = nn.Conv2d(64, 64, 5, padding=2)
        #
        self.conv_5_5 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv_5_6 = nn.Conv2d(128, 128, 5, padding=2)

        ## kernel = 7*7
        self.conv_7_1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conv_7_2 = nn.Conv2d(32, 32, 7, padding=3)
        #
        self.conv_7_3 = nn.Conv2d(32, 64, 7, padding=3)
        self.conv_7_4 = nn.Conv2d(64, 64, 7, padding=3)
        #
        self.conv_7_5 = nn.Conv2d(64, 128, 7, padding=3)
        self.conv_7_7 = nn.Conv2d(128, 128, 7, padding=3)

    def forward(self, x):
        size_in = x.size(0)

        ## kernel=3*3
        out_3 = self.conv_3_1(x)
        out_3 = self.conv_3_2(out_3)
        out_3 = F.max_pool2d(out_3)
        out_3 = F.relu(out_3)

        #
        out_3 = self.conv_3_3(out_3)
        out_3 = self.conv_3_4(out_3)
        out_3 = F.max_pool2d(out_3)
        out_3 = F.relu(out_3)

        #
        out_3 = self.conv_3_5(out_3)
        out_3 = self.conv_3_6(out_3)
        out_3 = F.max_pool2d(out_3)
        out_3 = F.relu(out_3)

        ## kernel=5*5
        out_5 = self.conv_5_1(x)
        out_5 = self.conv_5_2(out_5)
        out_5 = F.max_pool2d(out_5)
        out_5 = F.relu(out_5)   

        #
        out_5 = self.conv_5_3(out_5)
        out_5 = self.conv_5_4(out_5)
        out_5 = F.max_pool2d(out_5)
        out_5 = F.relu(out_5)

        #
        out_5 = self.conv_5_5(out_5)
        out_5 = self.conv_5_6(out_5)
        out_5 = F.max_pool2d(out_5)
        out_5 = F.relu(out_5)   
        
        ## kernel=7*7
        out_7 = self.conv_7_1(x)
        out_7 = self.conv_7_2(out_7)
        out_7 = F.max_pool2d(out_7)
        out_7 = F.relu(out_7)

        #
        out_7 = self.conv_7_3(out_7)
        out_7 = self.conv_7_4(out_7)
        out_7 = F.max_pool2d(out_7)
        out_7 = F.relu(out_7)

        #
        out_7 = self.conv_7_5(out_7)
        out_7 = self.conv_7_6(out_7)
        out_7 = F.max_pool2d(out_7)
        out_7 = F.relu(out_7)