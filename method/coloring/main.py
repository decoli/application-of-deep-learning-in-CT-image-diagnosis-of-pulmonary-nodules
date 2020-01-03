import argparse

from pre_processing.utility import get_image_info, world2voxel_coord


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', default=32)

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--rate-training', default=0.9, type=float)
    parser.add_argument('--size-batch', default=32, type=int)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--random-seed', default=None, type=int)

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    parser.add_argument('--out-csv', type=str, default='test')

    args = parser.parse_args()
    return args

args = argument()

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



def main():
    pass

if __name__ == '__main__':
    main()
