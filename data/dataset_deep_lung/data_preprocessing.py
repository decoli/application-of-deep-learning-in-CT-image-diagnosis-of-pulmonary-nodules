root_luna16 = '/Volumes/shirui_WD_2/lung_image/all_LUNA16/LUNA16'
path_annotation = 'data/dataset_deep_lung/annotationdetclssgm_doctor.csv'
path_annotation_shirui = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui.csv'

import pandas as pd
import glob
import os
import csv
import pprint

recursive_path_mhd = os.path.join(root_luna16, '**', '*.mhd')
list_path_mhd = glob.glob(recursive_path_mhd, recursive=True)

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