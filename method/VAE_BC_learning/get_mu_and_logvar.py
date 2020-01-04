import argparse
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
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
    parser.add_argument('--size-batch', type=int, default=1024)

    parser.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-visdom', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)

    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    args.device = device

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
    if not args.no_visdom or args.plot:
        scatter_center = np.array([0, 0])

        # size_batch >= len(list_train)
        if batch_idx == 0:

            for feature_idx in range(mu.shape[1]):

                list_mu = []
                list_logvar = []

                for sample_idx in range(mu.shape[0]):
                    list_mu.append(mu[sample_idx, feature_idx])
                    list_logvar.append(logvar[sample_idx, feature_idx])

                if not args.no_visdom:
                    visdom_scatter( # bug
                        vis=visdom, x=np.array([np.array(list_mu), scatter_center]), y=np.array(list_logvar), win=str(feature_idx), name='test-get')

                elif args.plot:
                    plt.scatter(
                        x=np.array(list_mu), y=np.array(list_logvar))
                    plt.show()

        # size_batch < len(list_train)
        else:
            pass

if __name__ == "__main__":
    # get argument
    args = argument()

    # vae model
    path_model_vae = os.path.join(os.getcwd(), args.path_model_vae)
    model_vae = VAE(args)

    if torch.cuda.is_available():
        model_vae.load_state_dict(torch.load(path_model_vae))
    else:
        model_vae.load_state_dict(torch.load(path_model_vae,  map_location=torch.device('cpu')))

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
    data_set = Dataset(args, model_vae, list_train)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=args.size_batch,
        shuffle=True,
        )

    # visdom instance
    use_cross = str(args.use_cross)
    visdom = Visdom(
        env='use_cross == {use_cross}'.format(use_cross=use_cross)
    )

    # get mu and logvar of train-set
    get_mu_and_logvar(args, model_vae, list_train, visdom)
