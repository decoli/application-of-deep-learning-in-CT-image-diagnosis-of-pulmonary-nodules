'''
融合先验知识（如医师标注的各结节征象），进行良恶性分类的模型
'''

import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

BATCH_SIZE=256
EPOCHS=2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
RATE_TRAIN = 0.8
root_image = 'data/dataset_deep_lung/data_sample/png'
path_annotation_v2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
pd_annotation = pd.read_csv(path_annotation_v2)
list_data = []

for index, each_annotation in pd_annotation.iterrows():
    characteristics_dic = {
        # 'diameter_mm': each_annotation['diameter_mm'],
        ##
        'index': each_annotation['index'],
        'subtlety': each_annotation['subtlety'],
        'internalStructure': each_annotation['internalStructure'],
        'calcification': each_annotation['calcification'],
        'sphericity': each_annotation['sphericity'],
        'margin': each_annotation['margin'],
        'lobulation': each_annotation['lobulation'],
        'spiculation': each_annotation['spiculation'],
        'texture': each_annotation['texture'],
        ##
        'malignant': each_annotation['malignant'],
    }
    list_data.append(characteristics_dic)

class DataTraining(data.Dataset):
    def __init__(self, list_data):
        self.list_data = list_data
    
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self, idx):
        current_item = self.list_data[idx]

        # image
        path_image = os.path.join(
            root_image,
            '{index}{image_format}'.format(
                index=current_item['index'], image_format='.png'))
        image = cv2.imread(path_image, flags=2)
        image = torch.Tensor(image)
        image = torch.unsqueeze(image, 0)

        # label
        label = current_item['malignant']
        return_label = np.array(label)

        return image, return_label

class DataTesting(data.Dataset):
    def __init__(self, list_data):
        self.list_data = list_data
    
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self, idx):
        current_item = self.list_data[idx]

        # image
        path_image = os.path.join(root_image, current_item['index'])
        image = cv2.imread(path_image)

        # label
        label = current_item['malignancy']
        return_label = np.array(lalbe)

        return image, return_label

class PriorKnowledgeNet(nn.Module):
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
        self.fc_all_1 = nn.Linear(1536, 256)
        self.fc_all_2 = nn.Linear(256, 2)

    def forward(self, x):
        size_in = x.size(0)

        ## kernel=3*3
        out_3 = self.conv_3_1(x)
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
        out_5 = self.conv_5_1(x)
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
        out_7 = self.conv_7_1(x)
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
        out_all = F.relu(out_all)

        out_all = self.fc_all_2(out_all)

        #
        return out_all

model = PriorKnowledgeNet()
print('ddd')
# get train and test data
num_training = int(len(list_data) * RATE_TRAIN)
list_data_training = list_data[: num_training]
list_data_testing = list_data[num_training: ]

data_training = DataTraining(list_data_training)
data_testing = DataTesting(list_data_testing)

data_loader_training = DataLoader(data_training, batch_size=BATCH_SIZE, shuffle=True)
data_loader_testing = DataLoader(data_testing, batch_size=BATCH_SIZE, shuffle=True)

# get model
model = PriorKnowledgeNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

####
for epoch in range(1, EPOCHS + 1):

    total_loss_training = 0
    total_acc_training = 0
    model.train()

    count_train = 0
    for characteristics, label in data_loader_training:

        count_train += 1
        print(count_train)

        # input data
        input_data = characteristics.to(dtype=torch.float, device=DEVICE)

        # label
        label = label.to(dtype=torch.long, device=DEVICE)
        label = torch.squeeze(label)

        # optimizer
        optimizer.zero_grad()

        # model predict
        output = model(input_data)

        # get loss
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss_training += loss.item()

        # get acc
        result = torch.max(output, 1)[1].numpy()
        total_acc_training += sum(result == label.data.numpy())

    acc_training = total_acc_training / len(list_data_training)
    loss_training = total_loss_training / len(list_data_training)

    print('training loss:')
    print(loss_training)

    print('training acc')
    print(acc_training)

    if epoch % 60 == 0:
        # 模型测试
        pass
        total_loss_testing = 0
        total_acc_testing = 0
        model.eval()

        with torch.no_grad():
            for characteristics, label in data_loader_testing:
                # input data
                input_data = characteristics.to(dtype=torch.float, device=DEVICE)

                # label
                label = label.to(dtype=torch.long, device=DEVICE)
                label = torch.squeeze(label)

                # model predict
                output = model(input_data)

                # get loss
                loss = criterion(output, label)
                total_loss_testing += loss.item()

                # get acc
                result = torch.max(output, 1)[1].numpy()
                total_acc_testing += sum(result ==label.data.numpy())

        acc_testing = total_acc_testing / len(list_data_testing)
        loss_testing = total_loss_testing / len(list_data_testing)
        print('testing loss:')
        print(loss_testing)
        print('testing acc:')
        print(acc_testing)
