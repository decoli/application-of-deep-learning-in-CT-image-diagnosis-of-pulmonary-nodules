import sys
import pprint
import numpy as np


# utility for LUNA16

def world2voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def get_coordinate(current_item):
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

    coordinate = {
        'coordinate_x': voxel_coord_y, # .csvのxとyが逆になっている
        'coordinate_y': voxel_coord_x,
        'coordinate_z': voxel_coord_z:,
    }
    return coordinate

def get_image_info(input_pd):
    list_data = []
    for each_data in input_pd.iterrows():
        path_seriesuid_folder = each_data[1]['path_seriesuid_folder']
        seriesuid = each_data[1]['seriesuid']
        _class = each_data[1]['class']

        coord_x = each_data[1]['coord_x']
        coord_y = each_data[1]['coord_y']
        coord_z = each_data[1]['coord_z']

        origin_x = each_data[1]['origin_x']
        origin_y = each_data[1]['origin_y']
        origin_z = each_data[1]['origin_z']

        spacing_x = each_data[1]['spacing_x']
        spacing_y = each_data[1]['spacing_y']
        spacing_z = each_data[1]['spacing_z']

        diameter = each_data[1]['diameter']

        if sys.platform == 'darwin':
            path_seriesuid_folder = path_seriesuid_folder.replace('\\', '/')

        dict_append = {
            'path_seriesuid_folder': path_seriesuid_folder,
            'seriesuid': seriesuid,
            'class': _class,

            'coord_x': coord_x,
            'coord_y': coord_y,
            'coord_z': coord_z,

            'origin_x': origin_x,
            'origin_y': origin_y,
            'origin_z': origin_z,

            'spacing_x': spacing_x,
            'spacing_y': spacing_y,
            'spacing_z': spacing_z,
            
            'diameter': diameter,
        }
        list_data.append(dict_append)
    return list_data