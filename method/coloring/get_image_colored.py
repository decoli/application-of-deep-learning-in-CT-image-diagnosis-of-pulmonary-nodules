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

def read_image(dir_image, image_current):
    image_coordinate = get_coordinate(image_current)

    list_image = []

    # get name of subset
    name_subset = os.path.basename(os.path.dirname(image_current['path_seriesuid_folder'])).split('_')[0] + '_tiff'

    # read image
    image_index = image_coordinate[2]
    path_image = os.path.join(
        dir_image,
        name_subset,
        image_current['seriesuid'],
        'whole_image',
        'whole_{image_index}.tiff'.format(image_index=int(image_index))
        )

    image = cv2.imread(path_image)
    cv2.imwrite(args.path_image_colored, image)
    # cut the image


if __name__ == '__main__':
    args = argument()

    # get image info
    info_luna16 = pd.read_csv(args.path_info, index_col=0)
    list_info_image = get_image_info(info_luna16)

    # read image, cut image
    image_current = random.choice(list_info_image)
    image = read_image(args.dir_image, image_current) 

    # test
    print('test')
