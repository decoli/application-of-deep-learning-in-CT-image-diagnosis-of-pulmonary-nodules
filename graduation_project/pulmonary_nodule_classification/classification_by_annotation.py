import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

BATCH_SIZE=1024
EPOCHS=2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
RATE_TRAIN = 0.8

path_annotation_v2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
pd_annotation = pd.read_csv(path_annotation_v2)
list_data = []

class DataTraining(data.Dataset):
    def __init__(self, list_data):
        self.list_data = list_data

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        current_item = self.list_data[idx]

        ## diameter_mm
        # diameter_mm = current_item['diameter_mm']

        ## subtlety
        list_subtlety = [0] * 5
        fractional, integer = math.modf(current_item['subtlety'])
        list_subtlety[int(integer) - 1] = 1

        if fractional > 0:
            list_subtlety[int(integer)] = 1

        ## internalStructure
        list_internalStructure = [0] * 4
        fractional, integer = math.modf(current_item['internalStructure'])
        list_internalStructure[int(integer) - 1] = 1

        if fractional > 0:
            list_internalStructure[int(integer)] = 1

        ## calcification
        list_calcification = [0] * 6
        fractional, integer = math.modf(current_item['calcification'])
        list_calcification[int(integer) - 1] = 1

        if fractional > 0:
            list_calcification[int(integer)] = 1

        ## sphericity
        list_sphericity = [0] * 5
        fractional, integer = math.modf(current_item['sphericity'])
        list_sphericity[int(integer) - 1] = 1

        if fractional > 0:
            list_sphericity[int(integer)] = 1

        ## margin
        list_margin = [0] * 5
        fractional, integer = math.modf(current_item['margin'])
        list_margin[int(integer) - 1] = 1

        if fractional > 0:
            list_margin[int(integer)] = 1
 
        ## lobulation
        list_lobulation = [0] * 5
        fractional, integer = math.modf(current_item['lobulation'])
        list_lobulation[int(integer) - 1] = 1

        if fractional > 0:
            list_lobulation[int(integer)] = 1

        ## spiculation
        list_spiculation = [0] * 5
        fractional, integer = math.modf(current_item['spiculation'])
        list_spiculation[int(integer) - 1] = 1

        if fractional > 0:
            list_spiculation[int(integer)] = 1

        ## texture
        list_texture = [0] * 5
        fractional, integer = math.modf(current_item['texture'])
        list_texture[int(integer) - 1] = 1

        if fractional > 0:
            list_texture[int(integer)] = 1

        ## return
        list_characteristics = []
        # list_characteristics.append(diameter_mm) # 长度不是语义特征
        list_characteristics.extend(list_subtlety)
        list_characteristics.extend(list_internalStructure)
        list_characteristics.extend(list_calcification)
        list_characteristics.extend(list_sphericity)
        list_characteristics.extend(list_margin)
        list_characteristics.extend(list_lobulation)
        list_characteristics.extend(list_spiculation)
        list_characteristics.extend(list_texture)
        return_characteristics = np.array(list_characteristics)

        ## malignant
        malignant = current_item['malignant']
        return_malignant = np.array([malignant])

        return return_characteristics, return_malignant

class DataTesting(data.Dataset):
    def __init__(self, list_data):
        self.list_data = list_data

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        current_item = self.list_data[idx]

        ## diameter_mm
        # diameter_mm = current_item['diameter_mm']

        ## subtlety
        list_subtlety = [0] * 5
        fractional, integer = math.modf(current_item['subtlety'])
        list_subtlety[int(integer) - 1] = 1

        if fractional > 0:
            list_subtlety[int(integer)] = 1

        ## internalStructure
        list_internalStructure = [0] * 4
        fractional, integer = math.modf(current_item['internalStructure'])
        list_internalStructure[int(integer) - 1] = 1

        if fractional > 0:
            list_internalStructure[int(integer)] = 1

        ## calcification
        list_calcification = [0] * 6
        fractional, integer = math.modf(current_item['calcification'])
        list_calcification[int(integer) - 1] = 1

        if fractional > 0:
            list_calcification[int(integer)] = 1

        ## sphericity
        list_sphericity = [0] * 5
        fractional, integer = math.modf(current_item['sphericity'])
        list_sphericity[int(integer) - 1] = 1

        if fractional > 0:
            list_sphericity[int(integer)] = 1

        ## margin
        list_margin = [0] * 5
        fractional, integer = math.modf(current_item['margin'])
        list_margin[int(integer) - 1] = 1

        if fractional > 0:
            list_margin[int(integer)] = 1
 
        ## lobulation
        list_lobulation = [0] * 5
        fractional, integer = math.modf(current_item['lobulation'])
        list_lobulation[int(integer) - 1] = 1

        if fractional > 0:
            list_lobulation[int(integer)] = 1

        ## spiculation
        list_spiculation = [0] * 5
        fractional, integer = math.modf(current_item['spiculation'])
        list_spiculation[int(integer) - 1] = 1

        if fractional > 0:
            list_spiculation[int(integer)] = 1

        ## texture
        list_texture = [0] * 5
        fractional, integer = math.modf(current_item['texture'])
        list_texture[int(integer) - 1] = 1

        if fractional > 0:
            list_texture[int(integer)] = 1

        ## return
        list_characteristics = []
        # list_characteristics.append(diameter_mm) # 长度不是语义特征
        list_characteristics.extend(list_subtlety)
        list_characteristics.extend(list_internalStructure)
        list_characteristics.extend(list_calcification)
        list_characteristics.extend(list_sphericity)
        list_characteristics.extend(list_margin)
        list_characteristics.extend(list_lobulation)
        list_characteristics.extend(list_spiculation)
        list_characteristics.extend(list_texture)
        return_characteristics = np.array(list_characteristics)

        ## malignant
        malignant = current_item['malignant']
        return_malignant = np.array([malignant])

        return return_characteristics, return_malignant

for index, each_annotation in pd_annotation.iterrows():
    characteristics_dic = {
        # 'diameter_mm': each_annotation['diameter_mm'],
        ##
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

class AnnotationNet(nn.Module):
    def __init__(self):
        super(AnnotationNet, self).__init__()

        self.fc_1 = nn.Linear(40, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 128)
        self.fc_4 = nn.Linear(128, 128)
        self.fc_5 = nn.Linear(128, 128)
        self.fc_6 = nn.Linear(128, 2)

    def forward(self, x):
        size_in = x.size(0) # batch size

        out = self.fc_1(x)
        out = F.relu(out)
        out = self.fc_2(out)
        out = F.relu(out)

        out = self.fc_3(out)
        out = F.relu(out)

        # out = F.dropout(out)

        out = self.fc_4(out)
        out_ = F.relu(out)

        out = F.dropout(out_)
        out = self.fc_6(out)
        return out, out_

# get train and test data
num_training = int(len(list_data) * RATE_TRAIN)
list_data_training = list_data[: num_training]
list_data_testing = list_data[num_training: ]

data_training = DataTraining(list_data_training)
data_testing = DataTesting(list_data_testing)

data_loader_training = DataLoader(data_training, batch_size=BATCH_SIZE, shuffle=True)
data_loader_testing = DataLoader(data_testing, batch_size=BATCH_SIZE, shuffle=True)

# get model
model = AnnotationNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

####
for epoch in range(1, EPOCHS + 1):

    total_loss_training = 0
    total_acc_training = 0
    model.train()

    for characteristics, label in data_loader_training:

        # input data
        input_data = characteristics.to(dtype=torch.float, device=DEVICE)

        # label
        label = label.to(dtype=torch.long, device=DEVICE)
        label = torch.squeeze(label)

        # optimizer
        optimizer.zero_grad()

        # model predict
        output, output_ = model(input_data)

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

    # 模型测试
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
            output, output_ = model(input_data)

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
