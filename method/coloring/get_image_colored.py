import argparse
import os
import pprint
import sys
import pandas as pd
import random

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

if __name__ == '__main__':
    args = argument()

    info_luna16 = pd.read_csv(args.path_info, index_col=0)
    list_info_image = get_image_info(info_luna16)

    current_image = random.choice(list_info_image)
    get_coordinate = get_coordinate(current_image)

    print('test')
