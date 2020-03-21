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
from visdom import Visdom
# append sys.path
sys.path.append(os.getcwd())
from utility.visdom import (visdom_acc, visdom_loss, visdom_roc_auc, visdom_se,
                            visdom_sp)
BATCH_SIZE=256
EPOCHS=150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
RATE_TRAIN = 0.8
root_image = 'data/dataset_deep_lung/data_sample/png'
path_annotation_v2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
pd_annotation = pd.read_csv(path_annotation_v2)
list_data = []

##
# if args.visdom:
visdom = Visdom(
    env='extracting_semantics')

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

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

        # CNN input
        # image
        path_image = os.path.join(
            root_image,
            '{index}{image_format}'.format(
                index=current_item['index'], image_format='.png'))
        image_original = cv2.imread(path_image, flags=2)
        image_copy_1 = image_original.copy()
        image_copy_2 = image_original.copy()
        # cv2.imwrite('image_oririnal.png', image_original)
        image_original = torch.Tensor(image_original)
        image_original = torch.unsqueeze(image_original, 0)

        # region grow
        seeds = [Point(15,15), Point(16,15), Point(15,16), Point(16,16)]
        mask_1 = regionGrow(image_copy_1, seeds, 10)
        mask_2 = regionGrow(image_copy_2, seeds, 20)
        # cv2.imwrite('mask_1.png', mask_1 * 255)
        # cv2.imwrite('mask_2.png', mask_2 * 255)

        # get image masked (and transfered to Tensor)
        image_copy_1[mask_1==0] = [0]
        # cv2.imwrite('image_copy_1.png', np.array(image_copy_1))
        image_1 = torch.Tensor(image_copy_1)
        image_1 = torch.unsqueeze(image_1, 0)

        image_copy_2[mask_2==0] = [0]
        # cv2.imwrite('image_copy_2.png', np.array(image_copy_2))
        image_2 = torch.Tensor(image_copy_2)
        image_2 = torch.unsqueeze(image_2, 0)

        ## CNN label
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

        return image_original, image_1, image_2,  return_characteristics

class DataTesting(data.Dataset):
    def __init__(self, list_data):
        self.list_data = list_data
    
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self, idx):
        current_item = self.list_data[idx]

        # CNN input
        # image
        path_image = os.path.join(
            root_image,
            '{index}{image_format}'.format(
                index=current_item['index'], image_format='.png'))
        image_original = cv2.imread(path_image, flags=2)
        image_copy_1 = image_original.copy()
        image_copy_2 = image_original.copy()
        # cv2.imwrite('image_oririnal.png', image_original)
        image_original = torch.Tensor(image_original)
        image_original = torch.unsqueeze(image_original, 0)

        # region grow
        seeds = [Point(15,15), Point(16,15), Point(15,16), Point(16,16)]
        mask_1 = regionGrow(image_copy_1, seeds, 10)
        mask_2 = regionGrow(image_copy_2, seeds, 20)
        # cv2.imwrite('mask_1.png', mask_1 * 255)
        # cv2.imwrite('mask_2.png', mask_2 * 255)

        # get image masked (and transfered to Tensor)
        image_copy_1[mask_1==0] = [0]
        # cv2.imwrite('image_copy_1.png', np.array(image_copy_1))
        image_1 = torch.Tensor(image_copy_1)
        image_1 = torch.unsqueeze(image_1, 0)

        image_copy_2[mask_2==0] = [0]
        # cv2.imwrite('image_copy_2.png', np.array(image_copy_2))
        image_2 = torch.Tensor(image_copy_2)
        image_2 = torch.unsqueeze(image_2, 0)

        ## CNN label
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

        return image_original, image_1, image_2,  return_characteristics

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

model = ExtractingSemanticsModel().to(DEVICE)
# get train and test data
num_training = int(len(list_data) * RATE_TRAIN)
list_data_training = list_data[: num_training]
list_data_testing = list_data[num_training: ]

data_training = DataTraining(list_data_training)
data_testing = DataTesting(list_data_testing)

data_loader_training = DataLoader(data_training, batch_size=BATCH_SIZE, shuffle=True)
data_loader_testing = DataLoader(data_testing, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

####
for epoch in range(1, EPOCHS + 1):

    total_loss_training = 0
    total_acc_training = 0
    model.train()

    count_train = 0
    for x_1, x_2, x_3, label in data_loader_training:

        count_train += 1
        print(count_train)

        # input data
        input_data_1 = x_1.to(dtype=torch.float, device=DEVICE)
        input_data_2 = x_2.to(dtype=torch.float, device=DEVICE)
        input_data_3 = x_3.to(dtype=torch.float, device=DEVICE)

        # label
        label = label.to(dtype=torch.float, device=DEVICE)

        # optimizer
        optimizer.zero_grad()

        # model predict
        output = model(input_data_1, input_data_2, input_data_3)
        output = torch.sigmoid(output)

        # get loss
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss_training += loss.item()

    loss_training = total_loss_training / len(list_data_training)

    # visdom
    visdom_loss(
        visdom, epoch, loss_training, win='loss', name='training')
    print('training loss:')
    print(loss_training)

torch.save(model.state_dict(), 'model_extracting_semantics.pt')