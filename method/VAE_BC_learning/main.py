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
from method.vae_bc_learning.model import CnnModel
from utility.model.auto_encoding_variational import VAE
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
    parser.add_argument('--size-cutting', default=32)
    parser.add_argument('--learning-rate', type=int, default=1e-4)

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

    parser.add_argument('--rate-bc-class', type=float, default=0)
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

        # generate the image
        z = self.model_vae.reparameterize(image_current[0], image_current[1])
        image = self.model_vae.decode(z)

        image = image.view(self.args.size_cutting, self.args.size_cutting)
        image = image.detach().numpy()

        if self.args.get_mid_product:
            path_image_mid_produc = os.path.join(
                os.getcwd(), 'method', 'vae_bc_learning', 'test',
                'image_mid_produc{image_format}'.format(image_format='.png'))
            cv2.imwrite(path_image_mid_produc, image * 255)

        # get the label
        label = image_current[2].detach().numpy()

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
            image = cv2.resize(image, (50, 50))
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

    # get path of model vae
    path_model_vae = os.path.join(os.getcwd(), args.path_model_vae)

    # load model vae
    model_vae = VAE(args)
    model_vae.load_state_dict(torch.load(path_model_vae, map_location=args.device))

    # get image info
    info_luna16 = pd.read_csv(args.path_input, index_col=0)
    list_info_image = get_image_info(info_luna16)
    random.shuffle(list_info_image)

    # how to validation
    if not(args.num_cross is None) and not (args.use_cross is None):
        list_train, list_test = cross_validation(args, list_info_image)
    else:
        list_train, list_test = rate_validation(args, list_info_image)

    #### get mu and logvar
    from method.vae_bc_learning.get_mu_and_logvar import get_mu_and_logvar
    from method.vae_bc_learning.get_mu_and_logvar import Dataset

    data_set = Dataset(args, model_vae, list_train)
    size_batch_get_mu_and_logvar = 1024
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=size_batch_get_mu_and_logvar,
        shuffle=True,
        )

    args.no_visdom = True
    mu, logvar, label = get_mu_and_logvar(args, model_vae, data_loader, visdom=None)

    # define date loader
    data_set_train = DatasetTrain(args, list(zip(mu, logvar, label)), model_vae)
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
    env = 'vae_bc_learning'
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
