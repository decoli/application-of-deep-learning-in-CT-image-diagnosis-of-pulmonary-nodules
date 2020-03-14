import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

BATCH_SIZE=512 #大概需要2G的显存
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

path_annotation_v2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
pd_annotation = pd.read_csv(path_annotation_v2)
list_data = []

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
        list_subtlety[integer - 1] = 1

        if fractional > 0:
            list_subtlety[integer] = 1

        ## internalStructure
        list_internalStructure = [0] * 4
        fractional, integer = math.modf(current_item['internalStructure'])
        list_internalStructure[integer - 1] = 1

        if fractional > 0:
            list_internalStructure[integer] = 1

        ## calcification
        list_calcification = [0] * 6
        fractional, integer = math.modf(current_item['calcification'])
        list_calcification[integer - 1] = 1

        if fractional > 0:
            list_calcification[integer] = 1

        ## sphericity
        list_sphericity = [0] * 5
        fractional, integer = math.modf(current_item['sphericity'])
        list_sphericity[integer - 1] = 1

        if fractional > 0:
            list_sphericity[integer] = 1

        ## margin
        list_margin = [0] * 5
        fractional, integer = math.modf(current_item['margin'])
        list_margin[integer - 1] = 1

        if fractional > 0:
            list_margin[integer] = 1
 
        ## lobulation
        list_lobulation = [0] * 5
        fractional, integer = math.modf(current_item['lobulation'])
        list_lobulation[integer - 1] = 1

        if fractional > 0:
            list_lobulation[integer] = 1

        ## spiculation
        list_spiculation = [0] * 5
        fractional, integer = math.modf(current_item['spiculation'])
        list_spiculation[integer - 1] = 1

        if fractional > 0:
            list_spiculation[integer] = 1

        ## texture
        list_texture = [0] * 5
        fractional, integer = math.modf(current_item['texture'])
        list_texture[integer - 1] = 1

        if fractional > 0:
            list_texture[integer] = 1

        ## return
        list_characteristics = []
        # list_characteristics.append(diameter_mm) # 长度不是语义特征
        list_characteristics.extend(list_subtlety)
        list_internalStructure.extend(list_internalStructure)
        list_calcification.extend(list_calcification)
        list_sphericity.extend(list_sphericity)
        list_characteristics.extend(list_margin)
        list_characteristics.extend(list_lobulation)
        list_characteristics.extend(list_spiculation)
        list_characteristics.extend(list_texture)
        return_characteristics = np.array(list_characteristics)

        ## malignant
        malignant = [current_item['malignant']]
        return_malignant = np.array(malignant)

        return return_characteristics, return_malignant

class AnnotationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(40, 20)
        self.fc_2 = nn.Linear(20, 10)
        self.fc_3 = nn.Linear(10, 2)

    def forward(self, x):
        size_in = x.size(0)

        out = self.fc_1(x)
        out = F.relu(out)

        out = self.fc_2(x)
        out = F.relu(out)

        out = nn.Dropout(out)
        out = self.fc_3(x)
        out = F.log_softmax(out)

        return out

model = AnnotationNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train():
    model.train()


def test():
    model.eval()


for eopch in range(1, EPOCHS + 1):
    train(model)
    test(model)
