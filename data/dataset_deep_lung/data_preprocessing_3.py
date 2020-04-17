path_annotation_shirui_1 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui.csv'
# path_annotation_shirui_2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
path_annotation_shirui_2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2_test.csv'
path_mapping_csv = 'data/dataset_deep_lung/LIDC-IDRI-mappingLUNA16.csv'

# root_lidc = '/Volumes/shirui_WD_2/lung_image/all_LIDC/LIDC-IDRI'
root_lidc = 'G:/lung_image/all_LIDC/LIDC-IDRI'

import csv
import glob
import os

import pandas as pd
from bs4 import BeautifulSoup

# get annotation_shirui
pd_annotation_1 = pd.read_csv(path_annotation_shirui_1)
pd_annotation_1.index += 1
# print(pd_annotation_1)

# get mapping
pd_mapping = pd.read_csv(path_mapping_csv)
pd_mapping.index += 1
# print(pd_mapping)

# search for the .xml file
count_no_file = 0
count_too_long = 0
count_small_nodule = 0
count = 0

with open(path_annotation_shirui_2, 'w') as f:
    writer = csv.writer(f)
    writer.writerow([
        'index',
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
        ##
        'subtlety',
        'internalStructure',
        'calcification',
        'sphericity',
        'margin',
        'lobulation',
        'spiculation',
        'texture',
        'malignancy',
        ])

for index, each_annotation in pd_annotation_1.iterrows():
    count += 1

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

    list_path_xml = glob.glob(glob_path_xml)
    if len(list_path_xml) == 0: # 文件夹中找不到.xml文件.
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
    with open(path_xml, 'r') as file_xml:
        markup = file_xml.read()
    xml = BeautifulSoup(markup, features='xml')

    reading_sessions = xml.LidcReadMessage.find_all('readingSession')
    list_dic = []
    for each_reading in reading_sessions:
        nodules = each_reading.find_all('unblindedReadNodule') # 每个 unblindedReadNodule 表示一个（大或小）结节

        flag_read = False
        for nodule in nodules:
            rois = nodule.find_all('roi')
            for each_roi in rois:
                condition__5 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] - 5)
                condition__4 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] - 4)
                condition__3 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] - 3)
                condition__2 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] - 2)
                condition__1 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] - 1)
                condition_0 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'])
                condition_1 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] + 1)
                condition_2 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] + 2)
                condition_3 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] + 3)
                condition_4 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] + 4)
                condition_5 = int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ'] + 5)

                if condition__1 or condition__2 or condition__3 or condition__4 or condition__5 or condition_0 or condition_1 or condition_2 or condition_3 or condition_4 or condition_5: # 应该有一个允许范围
                    if nodule.characteristics:
                        characteristics_dic = {
                            'subtlety': int(nodule.characteristics.subtlety.text),
                            'internalStructure': int(nodule.characteristics.internalStructure.text),
                            'calcification': int(nodule.characteristics.calcification.text),
                            'sphericity': int(nodule.characteristics.sphericity.text),
                            'margin': int(nodule.characteristics.margin.text),
                            'lobulation': int(nodule.characteristics.lobulation.text),
                            'spiculation': int(nodule.characteristics.spiculation.text),
                            'texture': int(nodule.characteristics.texture.text),
                            'malignancy': int(nodule.characteristics.malignancy.text),
                            }
                        list_dic.append(characteristics_dic)
                        flag_read = True
                        break

            if flag_read:
                break

    if len(list_dic) == 0:
        print('small nodule: {path_xml}'.format(path_xml=path_xml))
        count_small_nodule += 1
        continue
    
    # get mean value
    sum_subtlety = 0
    sum_internalStructure = 0
    sum_calcification = 0
    sum_sphericity  = 0
    sum_margin = 0
    sum_lobulation = 0
    sum_spiculation = 0
    sum_texture = 0
    sum_malignancy = 0
    for each_dic in list_dic:
        sum_subtlety += each_dic['subtlety']
        sum_internalStructure += each_dic['internalStructure']
        sum_calcification += each_dic['calcification']
        sum_sphericity += each_dic['sphericity']
        sum_margin += each_dic['margin']
        sum_lobulation += each_dic['lobulation']
        sum_spiculation += each_dic['spiculation']
        sum_texture += each_dic['texture']
        sum_malignancy += each_dic['malignancy']
    mean_subtlety = sum_subtlety / len(list_dic)
    mean_internalStructure = sum_internalStructure / len(list_dic)
    mean_calcification = sum_calcification / len(list_dic)
    mean_sphericity = sum_sphericity / len(list_dic)
    mean_margin = sum_margin / len(list_dic)
    mean_lobulation = sum_lobulation / len(list_dic)
    mean_spiculation = sum_spiculation / len(list_dic)
    mean_texture = sum_texture / len(list_dic)
    mean_malignancy = sum_malignancy / len(list_dic)

    print(count)

    # write into v2.csv
    writer_row = []
    ##
    writer_row.append(index)
    writer_row.append(each_annotation['seriesuid'])
    writer_row.append(each_annotation['coordX'])
    writer_row.append(each_annotation['coordY'])
    writer_row.append(each_annotation['coordZ'])
    ##
    writer_row.append(each_annotation['OffsetX'])
    writer_row.append(each_annotation['OffsetY'])
    writer_row.append(each_annotation['OffsetZ'])
    ##
    writer_row.append(each_annotation['ElementSpacingX'])
    writer_row.append(each_annotation['ElementSpacingY'])
    writer_row.append(each_annotation['ElementSpacingZ'])
    ##
    writer_row.append(each_annotation['diameter_mm'])
    writer_row.append(each_annotation['malignant'])
    ##
    writer_row.append(mean_subtlety)
    writer_row.append(mean_internalStructure)
    writer_row.append(mean_calcification)
    writer_row.append(mean_sphericity)
    writer_row.append(mean_margin)
    writer_row.append(mean_lobulation)
    writer_row.append(mean_spiculation)
    writer_row.append(mean_texture)
    writer_row.append(mean_malignancy)
    with open(path_annotation_shirui_2, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(writer_row)

print(count_no_file) # 共有88个样本找不到xml文件
