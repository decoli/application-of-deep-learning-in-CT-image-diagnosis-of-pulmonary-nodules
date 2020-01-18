# append sys.path
import argparse
import os
import random
import sys

del sys.path[0]
sys.path.append(os.getcwd())

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn, optim
from visdom import Visdom

from utility.model.cnn_simple import CnnSimple
from utility.pre_processing import (cross_validation, get_coordinate,
                                    get_image_info, rate_validation)
from utility.visdom import (
    visdom_acc, visdom_loss, visdom_roc_auc, visdom_se, visdom_sp)


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-input', type=str)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', type=int, default=32)
    parser.add_argument('--rate-learning', type=float, default=1e-4)
    parser.add_argument('--dropout', action='store_true', default=False)

    parser.add_argument('--size-batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)
    
    parser.add_argument('--visdom', action='store_true', default=False)

    args = parser.parse_args()

    # set device for pytorch
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    args.device = device

    # set random seed
    random.seed(args.seed)

    return args
# define data set
class DatasetTrain():
    def __init__(self, args, list_data_set):
        self.args = args
        self.list_data_set = list_data_set

    def __len__(self):
        return len(self.list_data_set)

    def __getitem__(self, idx):
        image_current = self.list_data_set[idx]
        image_coordinate = get_coordinate(image_current)

        # get image path
        name_subset = os.path.basename(
            os.path.dirname(image_current['path_seriesuid_folder'])
            ).split('_')[0] + '_tiff'
        image_index = int(image_coordinate['coordinate_z'])

        path_image = os.path.join(
            args.dir_image,
            name_subset,
            image_current['seriesuid'],
            'whole_image',
            'whole_{image_index}.tiff'.format(image_index=image_index)
            )
        image = cv2.imread(path_image, flags=2)
        image = image / 255

        # cut the image
        x_start = int(image_coordinate['coordinate_x'] - args.size_cutting / 2)
        x_end = int(image_coordinate['coordinate_x'] + args.size_cutting / 2)
        y_start = int(image_coordinate['coordinate_y'] - args.size_cutting / 2)
        y_end = int(image_coordinate['coordinate_y'] + args.size_cutting / 2)
        image = image[x_start: x_end, y_start: y_end]
        image = cv2.resize(image, (50, 50))
        image = np.expand_dims(image, 0)

        # get the label
        label = int(image_current['class'])
        if label == 0:
            label = np.array([0])
        elif label == 1:
            label = np.array([1])

        return image, label

class DatasetTest():
    def __init__(self, args, list_data_set):
        self.args = args
        self.list_data_set = list_data_set

    def __len__(self):
        return len(self.list_data_set)

    def __getitem__(self, idx):
        image_current = self.list_data_set[idx]
        image_coordinate = get_coordinate(image_current)

        # get image path
        name_subset = os.path.basename(
            os.path.dirname(image_current['path_seriesuid_folder'])
            ).split('_')[0] + '_tiff'
        image_index = int(image_coordinate['coordinate_z'])

        path_image = os.path.join(
            args.dir_image,
            name_subset,
            image_current['seriesuid'],
            'whole_image',
            'whole_{image_index}.tiff'.format(image_index=image_index)
            )
        image = cv2.imread(path_image, flags=2)
        image = image / 255

        # cut the image
        x_start = int(image_coordinate['coordinate_x'] - args.size_cutting / 2)
        x_end = int(image_coordinate['coordinate_x'] + args.size_cutting / 2)
        y_start = int(image_coordinate['coordinate_y'] - args.size_cutting / 2)
        y_end = int(image_coordinate['coordinate_y'] + args.size_cutting / 2)
        image = image[x_start: x_end, y_start: y_end]
        image = cv2.resize(image, (50, 50))
        image = np.expand_dims(image, 0)

        # get the label
        label = int(image_current['class'])
        if label == 0:
            label = np.array([0])
        elif label == 1:
            label = np.array([1])

        return image, label

def count_sample(list_sample, type_sample):
    count_benign = 0
    count_malignant = 0

    for each_sample in list_sample:
        if each_sample['class'] == 1: # opposite ?
            count_benign = count_benign + 1
        elif each_sample['class'] == 0: # opposite ?
            count_malignant = count_malignant + 1

    print(
        '---{type_sample}---\n'
        '---number of benign samples: {count_benign}\n'
        '---number of malignant samples: {count_malignant}\n'
        .format(
            type_sample=type_sample,
            count_benign=count_benign,
            count_malignant=count_malignant,
            )
        )

def log_batch(prediction, label, loss_batch, loss, tp, fn, fp, tn, prediction_list, label_list, type_batch):
    prediction = nn.functional.softmax(prediction, dim=1)

    # append probability list
    for (each_prediction, each_label) in zip(list(prediction), label):
        prediction_list.append(each_prediction[1].detach().cpu().numpy())
        label_list.append(each_label.detach().cpu().numpy())

    # compute tp, fn, fp, tn
    prediction = torch.round(prediction) # https://pytorch.org/docs/master/torch.html#math-operations

    for (each_prediction, each_label) in zip(list(prediction), label):
        if each_label == 1 and each_prediction[1] == 1:
            tp += 1
        if each_label == 1 and each_prediction[0] == 1:
            fn += 1
        if each_label == 0 and each_prediction[1] == 1:
            fp += 1
        if each_label == 0 and each_prediction[0] == 1:
            tn += 1

    loss = loss + loss_batch
    return loss, tp, fn, fp, tn, prediction_list, label_list

def log_epoch(epoch, loss, tp, fn, fp, tn, args, prediction_list, label_list, visdom, visdom_name, type_epoch):
    count_sample = tp + fn + fp + tn

    acc = (tp + tn) / count_sample
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    loss = loss / count_sample

    # compute roc auc
    roc_auc = roc_auc_score(np.array(label_list), np.array(prediction_list))

    # output visdom
    if args.visdom:
        visdom_acc(
            visdom, epoch, acc, win='acc', name=visdom_name)
        visdom_loss(
            visdom, epoch, loss, win='loss', name=visdom_name)
        visdom_se(
            visdom, epoch, se, win='se', name=visdom_name)
        visdom_sp(
            visdom, epoch, sp, win='sp', name=visdom_name)
        visdom_roc_auc(
            visdom, epoch, roc_auc, win='roc_auc', name=visdom_name)

def train(model, optimizer, criterion, train_loader, epoch, args, visdom):

    model.train()

    loss = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    # for making roc
    prediction_list = []
    label_list = []

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(args.device, dtype=torch.float)
        label = label.to(args.device, dtype=torch.long)

        # train the model
        optimizer.zero_grad()

        # model predict
        prediction = model(data)

        # get loss
        prediction = torch.squeeze(prediction)
        label = torch.squeeze(label)
        loss_batch = criterion(prediction, label) # https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # model step
        loss_batch.backward()
        optimizer.step()

        # log for each batch
        loss, tp, fn, fp, tn, prediction_list, label_list = log_batch(
            prediction, label, loss, loss_batch, tp, fn, fp, tn, prediction_list, label_list, type_batch='batch')

    # log for each epoch
    log_epoch(
        epoch, loss, tp, fn, fp, tn, args, prediction_list, label_list, visdom, visdom_name='train', type_epoch='train')

def test(model, test_loader, epoch, args, visdom):

    model.eval()

    loss = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    # for making roc
    prediction_list = []
    label_list = []

    # test the model
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(args.device, dtype= torch.float)
            label = label.to(args.device, dtype= torch.long)

            # model predict
            prediction = model(data)

            # get loss
            prediction = torch.squeeze(prediction)
            label = torch.squeeze(label)
            loss_batch = criterion(prediction, label) # https://pytorch.org/docs/stable/nn.html#crossentropyloss

            # log for each batch
            loss, tp, fn, fp, tn, prediction_list, label_list = log_batch(
                prediction, label, loss, loss_batch, tp, fn, fp, tn, prediction_list, label_list, type_batch='test')

    # log for each epoch
    log_epoch(
        epoch, loss, tp, fn, fp, tn, args, prediction_list, label_list, visdom, visdom_name='test', type_epoch='test')

if __name__ == '__main__':
    # get argument
    args = argument()

    # get image info
    info_luna16 = pd.read_csv(args.path_input, index_col=0)
    list_info_image = get_image_info(info_luna16)
    random.shuffle(list_info_image)

    # how to validation
    if not(args.num_cross is None) and not (args.use_cross is None):
        list_train, list_test = cross_validation(args, list_info_image)
    else:
        list_train, list_test = rate_validation(args, list_info_image)

    # define date loader
    data_set_train = DatasetTrain(args, list_train)
    data_set_test = DatasetTest(args, list_test)

    train_loader = torch.utils.data.DataLoader(
        data_set_train,
        batch_size=args.size_batch,
        shuffle=True,
        )
    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=args.size_batch,
        shuffle=True,
        )

    # model instance
    model = CnnSimple(args).to(args.device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.rate_learning)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # visdom instance
    env = 'cnn_simple'
    if args.visdom:
        visdom = Visdom(
            env=env)
    else:
        visdom = None

    # model train and test
    for epoch in range(1, args.epoch + 1):
        train(model, optimizer, criterion, train_loader, epoch, args, visdom)
        test(model, test_loader, epoch, args, visdom)

        print('---epoch: {epoch:>4}---'.format(epoch=epoch))
