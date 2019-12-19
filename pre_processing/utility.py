# utility for LUNA16

def world2voxel_coord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def get_image_info(input_pd):
    list_data = []
    for each_data in input_pd.iterrows():
        path_seriesuid_folder = each_data[1][0]
        seriesuid = each_data[1][1]
        each_class = each_data[1][6]

        coord_x = each_data[1][2]
        coord_y = each_data[1][3]
        coord_z = each_data[1][4]

        origin_x = each_data[1][8]
        origin_y = each_data[1][9]
        origin_z = each_data[1][10]

        spacing_x = each_data[1][11]
        spacing_y = each_data[1][12]
        spacing_z = each_data[1][13]

        diameter = each_data[1][5]

        if sys.platform == 'darwin':
            path_seriesuid_folder = path_seriesuid_folder.replace('\\', '/')

        dict_append = {
            'path_seriesuid_folder': path_seriesuid_folder,
            'seriesuid': seriesuid,
            'class': each_class,

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