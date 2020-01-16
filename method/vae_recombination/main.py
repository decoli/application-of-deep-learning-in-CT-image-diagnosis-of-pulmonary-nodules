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
import torch.nn.functional as nnf
import torch.utils.data
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn, optim
from torch.nn import functional as nnf
from torchvision import datasets, transforms
from torchvision.utils import save_image
from visdom import Visdom

# append sys.path
sys.path.append(os.getcwd())
from utility.model.auto_encoding_variational import VAE
from utility.model.cnn_simple import CnnSimple
from utility.pre_processing import (cross_validation, get_coordinate,
                                    get_image_info, rate_validation)
from utility.visdom import (
    visdom_acc, visdom_loss, visdom_roc_auc, visdom_se, visdom_sp)


# define argument
def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--path-model-vae',
        default=os.path.join(os.getcwd(), 'data', 'model', 'model_vae.pt'))
    parser.add_argument('--dimension-latent', type=int, default=20)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', type=int, default=32)
    parser.add_argument('--rate-learning', type=float, default=1e-4)
    parser.add_argument('--dropout', action='store_true', default=False)

    parser.add_argument('--random-switch', action='store_true', default=False)
    parser.add_argument('--dynamic-train-set', action='store_true', default=False)
    parser.add_argument('--between-class', action='store_true', default=False)
    parser.add_argument('--test-original', action='store_true', default=False)

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

    parser.add_argument('--visdom', action='store_true', default=False)
    parser.add_argument('--get-mid-product', action='store_true', default=False)
    parser.add_argument('--no-recombination', type=float, default=0)

    args = parser.parse_args()

    # set device for pytorch
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    args.device = device

    # set random seed
    random.seed(args.seed)

    # set train set rate
    args.train_set_se = 50
    args.train_set_sp = 50

    return args

# define data set
class DatasetTrain():
    def __init__(self, args, list_data_set, model_vae=None):
        self.args = args
        self.list_data_set = list_data_set

        list_benign = [] # class == 0
        list_malignant = [] # class == 1

        for each_date in list_data_set:
            if each_date[2] == 0:
                list_benign.append(each_date)
            elif each_date[2] == 1:
                list_malignant.append(each_date)

        self.list_benign = list_benign
        self.list_malignant = list_malignant

        self.model_vae = model_vae

    def __len__(self):
        return len(self.list_data_set)

    def __getitem__(self, idx):

        # get the label
        if not self.args.dynamic_train_set:
            label = random.choice([0, 1])
            label = np.array([label])
        else:
            label_se = [1] * self.args.train_set_se
            label_sp = [0] * self.args.train_set_sp
            label = []
            label.extend(label_se)
            label.extend(label_sp)
            label = random_switch.choice(label)
            label = np.array([label])

        # get the normal distribution parameter
        if label == 0:
            list_normal_distribution = random.sample(self.list_benign, 2)
        elif label == 1:
            list_normal_distribution = random.sample(self.list_malignant, 2)
        elif label == 0.5:
            list_normal_distribution = []
            sample_benign = random.sample(self.list_benign, 1)
            sample_malignant = random.sample(self.list_malignant, 1)

            list_normal_distribution.extend(sample_benign)
            list_normal_distribution.extend(sample_malignant)            

        # get list of mu, logvar
        list_mu = []
        list_logvar = []

        if self.args.random_switch:
            list_swich = []
            list_swich_r = ['r'] * int(args.dimension_latent / 2)
            list_swich_l = ['l'] * int(args.dimension_latent / 2)
            list_swich.extend(list_swich_r)
            list_swich.extend(list_swich_l)
            random.shuffle(list_swich)
            for i, each_swich in enumerate(list_swich):
                if each_swich == 'r':
                    list_mu.append(list_normal_distribution[0][0][i].cpu())
                    list_logvar.append(list_normal_distribution[0][1][i].cpu())
                elif each_swich == 'l':
                    list_mu.append(list_normal_distribution[1][0][i].cpu())
                    list_logvar.append(list_normal_distribution[1][1][i].cpu())

        elif not self.args.random_switch:
            for i in range(self.args.dimension_latent):
                if i % 2 == 0:
                    list_mu.append(list_normal_distribution[0][0][i].cpu())
                    list_logvar.append(list_normal_distribution[0][1][i].cpu())
                elif i % 2 == 1:
                    list_mu.append(list_normal_distribution[1][0][i].cpu())
                    list_logvar.append(list_normal_distribution[1][1][i].cpu())

        # generate the image
        z = self.model_vae.reparameterize(
            torch.from_numpy(np.array(list_mu)), torch.from_numpy(np.array(list_logvar)))
        image = self.model_vae.decode(z.to(device=self.args.device))

        image = image.view(self.args.size_cutting, self.args.size_cutting)
        image = image.cpu().detach().numpy()
        image = cv2.resize(image, (50, 50))

        if self.args.get_mid_product:
            path_image_mid_produc = os.path.join(
                os.getcwd(), 'method', 'vae_bc_learning', 'test',
                'image_mid_produc{image_format}'.format(image_format='.png'))
            cv2.imwrite(path_image_mid_produc, image * 255)
        image = np.expand_dims(image, 0)

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
        # image = cv2.resize(image, (50, 50))
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

    if args.dynamic_train_set:
        if se - sp > 5:
            args.train_set_se = args.train_set_se - 1
        if sp - se > 5:
            args.train_set_sp = args.train_set_sp - 1

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
            if not args.test_original:
                data = model_vae(data)
                data = data[0]
                data = data.view(data.shape[0], 1, args.size_cutting, args.size_cutting)

            data = nnf.interpolate(data, size=(50, 50), mode='bicubic', align_corners=False)
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
    model_vae.to(args.device)

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
    from method.vae_recombination.get_mu_and_logvar import Dataset
    from method.vae_recombination.get_mu_and_logvar import get_mu_and_logvar

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
    model = CnnSimple(args).to(args.device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.rate_learning)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # visdom instance
    env = 'vae_recombination'
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
