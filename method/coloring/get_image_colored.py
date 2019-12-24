import argparse
import os
import pprint
import sys
import pandas as pd
import random
import cv2

sys.path.append(os.getcwd())
from pre_processing.utility import get_image_info, get_coordinate


def argument():

    # default value
    default_path_info = os.path.join(os.getcwd(), 'data', 'info', 'info_luna16.csv')
    default_dir_image = os.path.join(os.getcwd(), 'data', 'image')
    default_path_image_colored = os.path.join(
        os.path.dirname(__file__), 'test', 'image_colored.{format_image}'.format(format_image='png'))

    # add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-info', default=default_path_info, type=str)
    parser.add_argument('--dir-image', default=default_dir_image, type=str)
    parser.add_argument('--size-cutting', default=32, type=int)
    parser.add_argument('--path-image-colored', default=default_path_image_colored, type=str)

    # parse argument
    args = parser.parse_args()
    return args

def read_image(args, image_current):
    image_coordinate = get_coordinate(image_current)

    list_image = []

    # get name of subset
    name_subset = os.path.basename(
        os.path.dirname(image_current['path_seriesuid_folder'])
        ).split('_')[0] + '_tiff'

    # read image
    bias_z = int(image_current['diameter'] / (2 * image_current['spacing_z']))
    image_index = int(image_coordinate[2])

    for idx in range(image_index - bias_z, image_index + bias_z + 1):
        path_image = os.path.join(
            args.dir_image,
            name_subset,
            image_current['seriesuid'],
            'whole_image',
            'whole_{idx}.tiff'.format(idx=idx)
            )

        image = cv2.imread(path_image, flags=2)
        cv2.imwrite(args.path_image_colored, image) # test

        # cut the image
        x_start = int(image_coordinate[1] - args.size_cutting / 2)
        x_end = int(image_coordinate[1] + args.size_cutting / 2)
        y_start = int(image_coordinate[0] - args.size_cutting / 2)
        y_end = int(image_coordinate[0] + args.size_cutting / 2)

        image = image[x_start: x_end, y_start: y_end]

        cv2.imwrite(args.path_image_colored, image) # test

        # append to the list
        list_image.append(image)

    return list_image

if __name__ == '__main__':
    args = argument()

    # get image info
    info_luna16 = pd.read_csv(args.path_info, index_col=0)
    list_info_image = get_image_info(info_luna16)

    # read image, cut image
    image_current = random.choice(list_info_image)
    image = read_image(args, image_current) 

    # test
    print('test')
