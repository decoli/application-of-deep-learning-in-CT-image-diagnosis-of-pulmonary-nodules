'''
1) get the attention area by VAE model
2) model cnn part is x-ception (, maybe)
'''
import argparse
import os
import pprint
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from visdom import Visdom

# append sys.path
sys.path.append(os.getcwd())
from utility.model.auto_encoding_variational import VAE
from utility.model.model_cnn_attention_area import CnnModel
from utility.pre_processing import (cross_validation, get_coordinate,
                                    get_image_info, rate_validation)
from utility.visdom import (visdom_acc, visdom_loss, visdom_roc_auc, visdom_se,
                            visdom_sp)


# define argument
def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--path-model-vae',
        default=os.path.join(os.getcwd(), 'data', 'model', 'model_vae.pt'))
    parser.add_argument('--dimension_latent', type=int, default=20)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', default=32, type=int)
    parser.add_argument('--size-resize', default=32, type=int)
    parser.add_argument('--learning-rate', default=1e-3)

    parser.add_argument('--rate-train', default=0.8, type=float)
    parser.add_argument('--size-batch', type=int, default=128,
        help='input batch size for train (default: 128)')
    parser.add_argument('--epoch', type=int, default=10,
        help='number of epoch to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='enables CUDA train')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    parser.add_argument('--triple', action='store_true', default=False)
    parser.add_argument('--no-attention-area', action='store_true', default=False)
    parser.add_argument('--visdom', action='store_true', default=False)

    parser.add_argument('--get-mid-product', action='store_true', default=False)

    args = parser.parse_args()
    cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")
    args.device = device

    return args

# define data set
class DatasetTrain():
    def __init__(self, args, list_data_set, model_vae=None):
        self.args = args
        self.list_data_set = list_data_set
        self.model_vae = model_vae

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

        # get image attentioned
        if not args.no_attention_area:
            image = get_image_attentioned(self.model_vae, image, args)

            # get image before input
            if args.get_mid_product:
                path_save = os.path.join(
                    os.getcwd(),'method', 'cnn_attention_area', 'test',
                    'test_image_before_input{image_format}'.format(image_format='.png'))
                
                np_zeros = np.zeros((1, self.args.size_resize, self.args.size_resize))
                image_before_input = np.concatenate([np_zeros, image])
                cv2.imwrite(path_save, np.transpose(image_before_input * 255, (1,2,0)))

        else:
            # resize the image
            # image = cv2.resize(image, (self.args.size_resize, self.args.size_resize))

            # get image before input
            if args.get_mid_product:
                path_save = os.path.join(
                    os.getcwd(),'method', 'cnn_attention_area', 'test',
                    'test_image_before_input{image_format}'.format(image_format='.png'))
                cv2.imwrite(path_save, np.transpose(image * 255, (1,2,0)))

            image = np.expand_dims(image, 0)

        # get the label
        label = int(image_current['class'])
        if label == 0:
            label = np.array([0])
        elif label == 1:
            label = np.array([1])

        return image, label

class DatasetTest():
    def __init__(self, args, list_data_set, model_vae=None):
        self.args = args
        self.list_data_set = list_data_set
        self.model_vae = model_vae

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

        # get image attentioned
        if not args.no_attention_area:
            image = get_image_attentioned(self.model_vae, image, args)
        else:
            # resize the image
            image = cv2.resize(image, (self.args.size_resize, self.args.size_resize))
            image = np.expand_dims(image, 0)

        # get the label
        label = int(image_current['class'])
        if label == 0:
            label = np.array([0])
        elif label == 1:
            label = np.array([1])

        return image, label

def log_batch(prediction, label, loss_batch, loss, tp, fn, fp, tn, prediction_list, label_list):
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

def log_epoch(epoch, loss, tp, fn, fp, tn, args, prediction_list, label_list, visdom, visdom_name):
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

def get_image_attentioned(model_vae, image, args):

    image_original = image

    # get attention area
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    attention_area = model_vae(torch.Tensor(image))[0]
    attention_area = attention_area.view(
        args.size_cutting, args.size_cutting).detach().numpy()

    # resize
    # image_original = cv2.resize(image_original, (args.size_resize, args.size_resize))
    # attention_area = cv2.resize(attention_area, (args.size_resize, args.size_resize))

    # get image attentioned
    if args.triple:

        # triple
        image_triple = image_original * attention_area
        image_attentioned = np.stack([image_original, attention_area, image_triple])
    
    else:
        image_attentioned = np.stack([image_original, attention_area])

    # output mid-product
    if args.get_mid_product:
        np_zeros = np.zeros(shape=image_original.shape)

        mid_product_original = np.stack([np_zeros, image_original, np_zeros]) # BGR
        path_save = os.path.join(
            os.getcwd(),'method', 'cnn_attention_area', 'test',
            'test_image_original{image_format}'.format(image_format='.png'))
        cv2.imwrite(path_save, np.transpose(mid_product_original * 255, (1,2,0)))

        mid_product_attention_area = np.stack([np_zeros, np_zeros, attention_area]) # BGR
        path_save = os.path.join(
            os.getcwd(),'method', 'cnn_attention_area', 'test',
            'test_attention_area{image_format}'.format(image_format='.png'))
        cv2.imwrite(path_save, np.transpose(mid_product_attention_area * 255, (1,2,0)))

        if not args.triple:
            mid_product_image_attentioned = np.stack([np_zeros, image_original, attention_area]) # BGR
        else:
            mid_product_image_attentioned = np.stack([image_original, attention_area, image_triple])

        path_save = os.path.join(
            os.getcwd(),'method', 'cnn_attention_area', 'test',
            'test_image_attentioned{image_format}'.format(image_format='.png'))
        cv2.imwrite(path_save, np.transpose(mid_product_image_attentioned * 255, (1,2,0)))

    return image_attentioned

def train(model, model_vae, optimizer, criterion, train_loader, epoch, args, visdom):

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
        data = data.to(args.device, dtype= torch.float)
        label = label.to(args.device, dtype= torch.long)

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
            prediction, label, loss, loss_batch, tp, fn, fp, tn, prediction_list, label_list)

    # log for each epoch
    log_epoch(
        epoch, loss, tp, fn, fp, tn, args, prediction_list, label_list, visdom, visdom_name='train')

def test(model, model_vae, test_loader, epoch, args, visdom):

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
                prediction, label, loss, loss_batch, tp, fn, fp, tn, prediction_list, label_list)

    # log for each epoch
    log_epoch(
        epoch, loss, tp, fn, fp, tn, args, prediction_list, label_list, visdom, visdom_name='test')

if __name__ == "__main__":
    # get argument
    args = argument()

    # get vae model
    if not args.no_attention_area:
        # get path of model vae
        path_model_vae = os.path.join(os.getcwd(), args.path_model_vae)

        # load model vae
        model_vae = VAE(args)
        model_vae.load_state_dict(torch.load(path_model_vae, map_location=args.device))
    else:
        model_vae = None

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
    data_set_train = DatasetTrain(args, list_train, model_vae)
    data_set_test = DatasetTest(args, list_test, model_vae)

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
    model = CnnModel(args).to(args.device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # visdom instance
    env = 'no_attention_area' if args.no_attention_area else 'attention_area'
    if args.visdom:
        visdom = Visdom(
            env=env)
    else:
        visdom = None

    # model train and test
    for epoch in range(1, args.epoch + 1):
        train(model, model_vae, optimizer, criterion, train_loader, epoch, args, visdom)
        test(model, model_vae, test_loader, epoch, args, visdom)

        print('---epoch: {epoch:>4}---'.format(epoch=epoch))
