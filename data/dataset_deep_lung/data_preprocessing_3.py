path_annotation_shirui_1 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui.csv'
path_annotation_shirui_2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'
path_mapping_csv = 'data/dataset_deep_lung/LIDC-IDRI-mappingLUNA16.csv'

root_lidc = '/Volumes/shirui_WD_2/lung_image/all_LIDC/LIDC-IDRI'

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

# serch for the .xml file
count_no_file = 0
count_too_long = 0
count = 0
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
    with open(path_xml, 'r') as file_xml:
        markup = file_xml.read()
    xml = BeautifulSoup(markup, features='xml')

    reading_sessions = xml.LidcReadMessage.find_all('readingSession')
    list_dic = []
    for each_reading in reading_sessions:
        nodules = each_reading.find_all('unblindedReadNodule') # 每个 unblindedReadNodule 表示一个（大或小）结节
        for nodule in nodules:
            rois = nodule.find_all('roi')
            for each_roi in rois:
                if int(float(each_roi.imageZposition.text)) == int(each_annotation['coordZ']):
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
                        continue
                    else:
                        continue
            continue
        if len(list_dic) == 0:
            print('small nodule: {path_xml}'.format(path_xml=path_xml))
        
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
