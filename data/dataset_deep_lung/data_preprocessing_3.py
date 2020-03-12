path_annotation_shirui_1 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui.csv'
path_annotation_shirui_2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
path_mapping_csv = 'data/dataset_deep_lung/LIDC-IDRI-mappingLUNA16.csv'

root_lidc = '/Volumes/shirui_WD_2/lung_image/all_LIDC/LIDC-IDRI'

import glob
import os

import pandas as pd

# get annotation_shirui
pd_annotation_1 = pd.read_csv(path_annotation_shirui_1)
pd_annotation_1.index += 1
# print(pd_annotation_1)

# get mapping
pd_mapping = pd.read_csv(path_mapping_csv)
pd_mapping.index += 1
# print(pd_mapping)

# serch for the .xml file
count_no_file = 0
count_too_long = 0
for index, each_annotation in pd_annotation_1.iterrows():
    # print(each_annotation)
    seriesuid = each_annotation['seriesuid']
    # print(seriesuid)

    pd_case = pd_mapping[pd_mapping['SeriesInstanceUID'] == seriesuid]
    patient_id = pd_case.iat[0, 0] #['PatientID']
    study_instance_uid = pd_case.iat[0, 1] #['StudyInstanceUID']
    series_instance_uid = pd_case.iat[0, 2] #['SeriesInstanceUID']

    glob_path_xml = os.path.join(
        root_lidc,
        patient_id,
        study_instance_uid,
        series_instance_uid,
        '*.xml',
        )

    list_path_xml = glob.glob(glob_path_xml) # 文件夹中找不到.xml文件
    if len(list_path_xml) == 0:
        # print('no file')
        # print(glob_path_xml)
        count_no_file += 1
        continue

    # if len(list_path_xml) > 1: # 一个文件夹中有多个.xml文件时作记录。
    #     print('len too long')
    #     print(list_path_xml)
    #     count_too_long += 1
    #     continue
    # path_xml = list_path_xml[0]

    # .xml文件中的医师标注处理

# print('no file: {count_no_file}'.format(count_no_file=count_no_file))
# print('too long: {count_too_long}'.format(count_too_long=count_too_long))

    # map the date (800 of cases)
    path_xml = list_path_xml[0]
