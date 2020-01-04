import argparse
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from visdom import Visdom
sys.path.append(os.getcwd())
from utility.auto_encoding_variational import VAE
from utility.pre_processing import (cross_validation, get_coordinate,
                                    get_image_info, rate_validation)
from utility.visdom import visdom_scatter


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--path-model-vae',
        default=os.path.join(os.getcwd(), 'utility', 'model', 'model_vae.pt'))
    parser.add_argument('--dimension-latent', type=int, default=20)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=1024)

    parser.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    parser.add_argument('--no-visdom', action='store_true', default=False)

    args = parser.parse_args()
    return args

class Dataset():
    def __init__(self, args, model_vae, list_data_set):
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
        image = np.expand_dims(image, 0)
        return image

def get_mu_and_logvar(args, model_vae, list_train, visdom):

    list_mu = []
    list_logvar = []

    model_vae.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data.to(args.device, dtype= torch.float)

            # get mu, logvar
            prediction, mu, logvar = model_vae(data)
            list_mu.append(mu)
            list_logvar.append(logvar)

    # output visdom
    if not args.no_visdom:
        print('test')

if __name__ == "__main__":
    # get argument
    args = argument()

    # vae model
    path_model_vae = os.path.join(os.getcwd(), args.path_model_vae)
    model_vae = VAE(args)
    model_vae.load_state_dict(torch.load(path_model_vae))

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
    data_set = Dataset(args, list_train, model_vae)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=args.size_batch,
        shuffle=True,
        )

    # visdom instance
    env = 'mu_and_logvar'
    visdom = Visdom(
        env=env
    )

    # get mu and logvar of train-set
    get_mu_and_logvar(args, model_vae, list_train, visdom)
