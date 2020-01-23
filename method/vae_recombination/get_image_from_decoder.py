'''
get the image from devoder of different epoch
original image set is 'train set' or 'test set'

step.1
train the vae model. (set the epoch)

model path:
data\\model\\model_vae_random_*_cross_*_epoch_*.pt

step.2
select the image from 'train set' or 'test set' (important)
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
from utility.model.auto_encoding_variational import VAE
from utility.pre_processing import (cross_validation, get_coordinate,
                                    get_image_info, rate_validation)


# define argument
def argument():
    parser = argparse.ArgumentParser(description='VAE for utility')
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', type=int, default=32)
    parser.add_argument('--dimension-latent', type=int, default=20)

    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='enables CUDA train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging train status')

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    # set the number within the 'train/test set'
    # then feed the vae model with the image seleceted by the number
    parser.add_argument('--single', action='store_true', default=None)
    parser.add_argument('--number-image-input', type=int, default=None)

    # randomly seclet two images to generate the breeding image
    parser.add_argument('--breeding', action='store_true', default=None)

    parser.add_argument('--path-load-model', type=str, default=None)
    parser.add_argument('--set-for-decoder', type=str, required=True, default=None)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return args, device, kwargs

def check_argument(args, list_to_select):

    # sys.exit()

    if not args.number_image_input:
        print(
            '[--number-image-input]: the number within the "test set".\n'
            'feed the vae model with the image seleceted by the number.\n'
            '------'
            )
        sys.exit()

    if not args.path_load_model:
        print(
            '[--path-load-model]: the path of the vae model.\n'
            '------'
            )
        sys.exit()

    if args.number_image_input + 1 > len(list_to_select):
        print(
            'the length of the list to select is : {len_list}\n'
            '------'
            .format(len_list=len(list_to_select) - 1)
            )
        sys.exit()

    # not sys.exit()

    if not args.number_image_input + 1 > len(list_to_select):
        print(
            'your selected number: {number_image_input}\n'
            'the max number can be: {len_list_to_select}'
            .format(
                number_image_input=args.number_image_input,
                len_list_to_select=len(list_to_select) - 1,
                )
            )

def get_image_from_decoder(args, device, list_to_select):
    # load the vae model
    model_vae = VAE(args)
    model_vae.load_state_dict(torch.load(args.path_load_model))
    model_vae = model_vae.to(device)

    # get image path
    image_current = list_to_select[args.number_image_input]
    image_coordinate = get_coordinate(image_current)

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

    # save the original image
    path_image_original = os.path.join(
        os.getcwd(),'method', 'vae_recombination', 'test', 'image_original{format_image}'.format(format_image='.png')
    )
    cv2.imwrite(path_image_original, image * 255)
    print(
        'original image saved: {path_image_original}'
        .format(path_image_original=path_image_original)
    )

    # feed the VAE image with the seleceted image
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    image = torch.Tensor(image)
    image = image.to(device)

    image_decoder = model_vae(image)[0]

    # save the image from decoder of VAE
    image_decoder = image_decoder.view(args.size_cutting, args.size_cutting)
    image_decoder = image_decoder.cpu().detach().numpy()

    path_image_decoder = os.path.join(
        os.getcwd(),'method', 'vae_recombination', 'test', 'image_decoder{format_image}'.format(format_image='.png')
    )
    cv2.imwrite(path_image_decoder, image_decoder * 255)
    print(
        'decoder image saved: {path_image_decoder}'
        .format(path_image_decoder=path_image_decoder)
    )

def get_image_breeding(args, device, list_data_set):
    # re-get image list
    list_benign = [] # class == 0
    list_malignant = [] # class == 1

    for each_date in list_data_set:
        if each_date[2] == 0:
            list_benign.append(each_date)
        elif each_date[2] == 1:
            list_malignant.append(each_date)

    # get the label
    label = random.choice([0, 1])
    label = np.array([label])

    # get the normal distribution parameter
    if label == 0:
        list_normal_distribution = random.sample(self.list_benign, 2)
    elif label == 1:
        list_normal_distribution = random.sample(self.list_malignant, 2)         

    # get list of mu, logvar
    list_mu = []
    list_logvar = []

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

    # save the image breeding
    path_image_mid_produc = os.path.join(
        os.getcwd(), 'method', 'vae_recombination', 'test',
        'image_breeding{image_format}'.format(image_format='.png'))
    cv2.imwrite(path_image_mid_produc, image * 255)

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

    # get image from decoder
    print('------')

    if args.single:
        if args.set_for_decoder == 'train': # select which set to decoder
            check_argument(args, list_train)
            get_image_from_decoder(args, device, list_train)

        elif args.set_for_devoder == 'test': # select which set to decoder
            check_argument(args, list_test)
            get_image_from_decoder(args, device, list_test)

    if args.breeding:
        # only for 'train' set image
        get_image_breeding(args, device, list_train)

    print('------')
