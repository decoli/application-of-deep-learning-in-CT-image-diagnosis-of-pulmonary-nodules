root_luna16 = '/Volumes/shirui_WD_2/lung_image/all_LUNA16/LUNA16'
path_annotation = 'data/dataset_deep_lung/annotationdetclssgm_doctor.csv'
path_annotation_shirui = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui.csv'

import csv
import glob
import os
import pprint

import numpy as np
import pandas as pd
import SimpleITK as sitk

recursive_path_mhd = os.path.join(root_luna16, '**', '*.mhd')
list_path_mhd = glob.glob(recursive_path_mhd, recursive=True)

def mhd_parser(path_mhd):

    itkimage = sitk.ReadImage(path_mhd)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    # ct_scan = sitk.GetArrayFromImage(itkimage)

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

    return offset_x, offset_y, offset_z, spacing_x, spacing_y, spacing_z

with open(path_annotation_shirui, 'w') as f:
    writer = csv.writer(f)
    writer.writerow([
        'seriesuid',
        ##
        'coordX',
        'coordY',
        'coordZ',
        ##
        'OffsetX',
        'OffsetY',
        'OffsetZ',
        ##
        'ElementSpacingX',
        'ElementSpacingY',
        'ElementSpacingZ',
        ##
        'diameter_mm',
        'malignant',
        ])

with open(path_annotation) as f:
    reader = csv.reader(f)
    for row in reader:
        for each_path_mhd in list_path_mhd:
            if row[0] == os.path.splitext(os.path.basename(each_path_mhd))[0]:

                # 解析mhd
                offset_x, offset_y, offset_z, spacing_x, spacing_y, spacing_z = mhd_parser(each_path_mhd)

                # 追加csv标注
                writer_row = []
                ##
                writer_row.append(row[0])
                ##
                writer_row.append(row[1])
                writer_row.append(row[2])
                writer_row.append(row[3])
                ##
                writer_row.append(offset_x)
                writer_row.append(offset_y)
                writer_row.append(offset_z)
                ##
                writer_row.append(spacing_x)
                writer_row.append(spacing_y)
                writer_row.append(spacing_z)
                ##
                writer_row.append(row[4])
                writer_row.append(row[5])

                with open(path_annotation_shirui, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(writer_row)
