'''
get the image from devoder of different epoch
original image set is 'train set'

step.1
train the vae model. (set the epoch)

model path:
data\\model\\model_vae_random_*_cross_*_epoch_*.pt

step.2
select the image from 'train set' (important)
feed the vae and get the image from the decoder.
'''

from __future__ import print_function

import argparse
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

#d append sys.path
sys.path.append(os.getcwd())
from utility.pre_processing import (cross_validation, get_coordinate,
                                    get_image_info, rate_validation)


# define argument
def argument():
    parser = argparse.ArgumentParser(description='VAE for utility')
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', type=int, default=28)
    parser.add_argument('--dimension-latent', type=int, default=20)

    parser.add_argument('--size-batch', type=int, default=128, metavar='N',
        help='input batch size for train (default: 128)')
    parser.add_argument('--epoch', type=int, default=10, metavar='N',
        help='number of epoch to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='enables CUDA train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging train status')

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    parser.add_argument('--path-save-model', default=os.path.join(os.getcwd(), 'data/model/model_vae.pt'), type=str,
        help='set the path of model to save')

    parser.add_argument('--path-image-input', type=str)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return args, device, kwargs

# define class 
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        image_pixel = int(args.size_cutting * args.size_cutting)

        self.fc1 = nn.Linear(image_pixel, 400)
        self.fc21 = nn.Linear(400, args.dimension_latent)
        self.fc22 = nn.Linear(400, args.dimension_latent)
        self.fc3 = nn.Linear(args.dimension_latent, 400)
        self.fc4 = nn.Linear(400, image_pixel)

        self.args = args

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.args.size_cutting * self.args.size_cutting))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, args):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, args.size_cutting * args.size_cutting), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# define data set
class DatasetTrain():
    def __init__(self, list_data_set, args):
        self.list_data_set = list_data_set
        self.args = args

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
            self.args.dir_image,
            name_subset,
            image_current['seriesuid'],
            'whole_image',
            'whole_{image_index}.tiff'.format(image_index=image_index)
            )
        image = cv2.imread(path_image, flags=2)
        image = image / 255

        # cut the image
        x_start = int(image_coordinate['coordinate_x'] - self.args.size_cutting / 2)
        x_end = int(image_coordinate['coordinate_x'] + self.args.size_cutting / 2)
        y_start = int(image_coordinate['coordinate_y'] - self.args.size_cutting / 2)
        y_end = int(image_coordinate['coordinate_y'] + self.args.size_cutting / 2)

        image = image[x_start: x_end, y_start: y_end]
        image = np.expand_dims(image, 0)
        return image

class DatasetTest():
    def __init__(self, list_data_set, args):
        self.list_data_set = list_data_set
        self.args = args

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
            self.args.dir_image,
            name_subset,
            image_current['seriesuid'],
            'whole_image',
            'whole_{image_index}.tiff'.format(image_index=image_index)
            )
        image = cv2.imread(path_image, flags=2)
        image = image / 255

        # cut the image
        x_start = int(image_coordinate['coordinate_x'] - self.args.size_cutting / 2)
        x_end = int(image_coordinate['coordinate_x'] + self.args.size_cutting / 2)
        y_start = int(image_coordinate['coordinate_y'] - self.args.size_cutting / 2)
        y_end = int(image_coordinate['coordinate_y'] + self.args.size_cutting / 2)

        image = image[x_start: x_end, y_start: y_end]
        image = np.expand_dims(image, 0)
        return image

def train(model, train_loader, epoch, device, args):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device, dtype= torch.float)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, args)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

if __name__ == "__main__":
    # get argument
    args, device, kwargs = argument()

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
    data_set_train = DatasetTrain(list_train, args)
    data_set_test = DatasetTest(list_test, args)

    train_loader = torch.utils.data.DataLoader(
        data_set_train,
        batch_size=args.size_batch,
        shuffle=True,
        **kwargs,
        )
    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=args.size_batch,
        shuffle=True,
        **kwargs,
        )

    # model instance
    model = VAE(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epoch + 1):
        train(model, train_loader, epoch, device, args)
    
    torch.save(model.state_dict(), args.path_save_model)
