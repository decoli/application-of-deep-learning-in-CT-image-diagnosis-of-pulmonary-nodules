path_annotation = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui.csv'
root_luna16 = '/Volumes/shirui_WD_2/lung_image/all_LUNA16/LUNA16'
root_save_npy = 'data/dataset_deep_lung/data_sample/npy'
root_save_png = 'data/dataset_deep_lung/data_sample/png'

s = 32 # size_cutting
s = int(s / 2 + 0.5)

import glob
import os

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk


'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_itk(path_mhd):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(path_mhd)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    offset_x = origin[2]
    offset_y = origin[1]
    offset_z = origin[0]
    spacing_x = spacing[2]
    spacing_y = spacing[1]
    spacing_z = spacing[0]

    return ct_scan, offset_x, offset_y, offset_z, spacing_x, spacing_y, spacing_z

def world2voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def normalizePlanes(npzarray):
     
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

annotation_pd = pd.read_csv(path_annotation)
annotation_pd.index += 1
print(annotation_pd)
for index, each_annotation in annotation_pd.iterrows():
    file_mhd = each_annotation['seriesuid'] + '.mhd'
    path_mhd = os.path.join(root_luna16, '**', file_mhd)
    path_mhd = glob.glob(path_mhd, recursive=True)
    path_mhd = path_mhd[0]

    ct_scan, offset_x, offset_y, offset_z, spacing_x, spacing_y, spacing_z = load_itk(path_mhd)
    ct_scan = normalizePlanes(ct_scan)

    x = world2voxel_coord(
        each_annotation['coordX'],
        offset_x,
        spacing_x,
        )
    x = int(x + 0.5)

    y = world2voxel_coord(
        each_annotation['coordY'],
        offset_y,
        spacing_y,
        )
    y = int(y + 0.5)

    z = world2voxel_coord(
        each_annotation['coordZ'],
        offset_z,
        spacing_z,
        )
    z = int(z + 0.5)
    
    location_nodule = ct_scan[z][y - s: y + s, x - s: x + s]

    # save .png
    name_png = '{index}.png'.format(index=index)
    path_png = os.path.join(root_save_png, name_png)
    cv2.imwrite(path_png, location_nodule * 255)

    # save .numpy
    name_npy = '{index}.npy'.format(index=index)
    path_png = os.path.join(root_save_npy, name_npy)
    np.save(path_png, location_nodule)