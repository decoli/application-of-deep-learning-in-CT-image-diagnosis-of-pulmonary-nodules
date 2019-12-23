import argparse
import os
import pprint
import sys

sys.path.append(os.getcwd())
from pre_processing.utility import get_image_info, world2voxel_coord


def argument():

    default_path_info = os.path.join(os.getcwd(), 'data', 'info', 'info_luna16.csv')
    default_dir_image = os.path.join(os.getcwd(), 'data', 'image')
    default_path_image_colored = os.path.join(
        os.path.dirname(__file__), 'test', 'image_colored.{format_image}'.format(format_image='png'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-info', default=default_path_info, type=str)
    parser.add_argument('--dir-image', default=default_dir_image, type=str)
    parser.add_argument('--size-cutting', default=32)

    parser.add_argument('--path-image-colored', default=default_path_image_colored, type=str)

    args = parser.parse_args()
    return args

def get_coordinate(list_image):
    voxel_coord_x = world2voxel_coord(
        current_item['coord_x'],
        current_item['origin_x'],
        current_item['spacing_x'],
        )
    voxel_coord_y = world2voxel_coord(
        current_item['coord_y'],
        current_item['origin_y'],
        current_item['spacing_y'],
        )
    voxel_coord_z = world2voxel_coord(
        current_item['coord_z'],
        current_item['origin_z'],
        current_item['spacing_z'],
        )
    voxel_coord_x = round(voxel_coord_x)
    voxel_coord_y = round(voxel_coord_y)
    voxel_coord_z = round(voxel_coord_z)

if __name__ == '__main__':
    args = argument()
    print('test')
