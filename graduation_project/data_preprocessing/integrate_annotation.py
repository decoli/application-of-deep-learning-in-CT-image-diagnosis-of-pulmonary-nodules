path_list_3_2 = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/list3.2.csv'
root_lidc = '/Volumes/shirui_WD_2/lung_image/LIDC-IDRI'
test_root_lidc = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/image/LIDC-IDRI'

import csv
import glob
import math
import os

import pandas as pd
import pydicom
from bs4 import BeautifulSoup

# read list_3_2
pd_3_2 = pd.read_csv(path_list_3_2)
pd_3_2.index += 1
print(pd_3_2)

recursive_path_xml = os.path.join(test_root_lidc, '**', '*.xml')
list_path_xml = glob.glob(recursive_path_xml, recursive=True)

for each_path_xml in list_path_xml:
    lidc_no = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(each_path_xml))))
    lidc_sub = os.path.basename(os.path.dirname(os.path.dirname(each_path_xml)))
    lidc_sub_sub = os.path.basename(os.path.dirname(each_path_xml))
    name_xml = os.path.basename(each_path_xml)

    # get dcm
    path_dicom = os.path.join(test_root_lidc, lidc_no, lidc_sub, lidc_sub_sub, '000001.dcm')
    dicom = pydicom.read_file(path_dicom)
    series_number = dicom.SeriesNumber

    # serch series number in list_3_2
    pd_scan = pd_3_2[pd_3_2['scan'] == int(series_number)]
    if pd_scan.empty:
        continue
    print(pd_scan)

    # parse xml
    with open(each_path_xml, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")

    # 在同一组结节中查询被标注的结节
    for each_scan in pd_scan.itertuples():
        if not math.isnan(each_scan['Unnamed: 13']): # 诊断太多， 大于医师人数
            continue
        if math.isnan(each_scan['Unnamed: 11']): # 诊断太少， 至少3个医师的诊断
        continue

        # get nodule IDs
        list_nodule_id = []
        list_nodule_id.append(each_scan['nodIDs']) # 第1个诊断
        list_nodule_id.append(each_scan['Unnamed: 10']) # 第2个诊断
        list_nodule_id.append(each_scan['Unnamed: 11']) # 第3个诊断
        if not math.isnan(each_scan['Unnamed: 12']): # 可能第4个诊断为空
            list_nodule_id.append(each_scan['Unnamed: 12']) # 第4个诊断

        # parsing diagnostic information
        flag_malignant_physician_1 = 0
        flag_malignant_physician_2 = 0
        flag_malignant_physician_3 = 0
        flag_malignant_physician_4 = 0

        flag_benign_physician_1 = 0
        flag_benign_physician_2 = 0
        flag_benign_physician_3 = 0
        flag_benign_physician_4 = 0

        reading_sessions = xml.LidcReadMessage.find_all('readingSession')

        for diagnostic_index, reading_session in enumerate(reading_sessions, 1): # 循环不同医师的诊断结果
            nodules = reading_session.find_all("unblindedReadNodule")

            for nodule in nodules:
                if nodule.characteristics:
                    # 判定1.结节被诊断医师人数 >= 3； 判定2.结节良恶性。
                    nodule_id = nodule.noduleID.text
                    malignancy = int(nodule.characteristics.malignancy.text)
                    if (not nodule_id in list_nodule_id) or malignancy == 3:
                        continue

                    if malignancy > 3: # 判定为恶性
                        if diagnostic_index == 1:
                            flag_malignant_physician_1 += 1
                            dic_characteristics_1 = get_dic_characteristics(nodule)

                        elif diagnostic_index == 2:
                            flag_malignant_physician_2 += 1
                            dic_characteristics_2 = get_dic_characteristics(nodule)

                        elif diagnostic_index == 3:
                            flag_malignant_physician_3 += 1
                            dic_characteristics_3 = get_dic_characteristics(nodule)
                            
                        elif diagnostic_index == 4:
                            flag_malignant_physician_4 += 1
                            dic_characteristics_4 = get_dic_characteristics(nodule)
                        
                    if malignancy < 3: # 判定为良性
                        if diagnostic_index == 1:
                            flag_benign_physician_1 += 1
                            dic_characteristics_1 = get_dic_characteristics(nodule)

                        elif diagnostic_index == 2:
                            flag_benign_physician_2 += 1
                            dic_characteristics_2 = get_dic_characteristics(nodule)

                        elif diagnostic_index == 3:
                            flag_benign_physician_3 += 1
                            dic_characteristics_3 = get_dic_characteristics(nodule)
                            
                        elif diagnostic_index == 4:
                            flag_benign_physician_4 += 1
                            dic_characteristics_4 = get_dic_characteristics(nodule)
        
        count_malignant_physician = flag_malignant_physician_1 + flag_malignant_physician_2 + flag_malignant_physician_3 + flag_malignant_physician_4
        count_benign_physician = flag_benign_physician_1 + flag_benign_physician_2 + flag_benign_physician_3 + flag_benign_physician_4

        if count_malignant_physician >= 3:
            # 判定为恶性
            class_malignant = 1

        if count_benign_physician >= 3:
            # 判定为良性
            class_malignant = 0

        if count_malignant_physician + count_benign_physician == 3:
            mean_dic = get_mean_dic_3(dic_characteristics_1, dic_characteristics_2, dic_characteristics_3)
        if count_malignant_physician + count_benign_physician == 4:
            mean_dic = get_mean_dic_4(dic_characteristics_1, dic_characteristics_2, dic_characteristics_3, dic_characteristics_4)

        # write info at new_annotation.csv 
        csv.writer(
            class_malignant
            mean_dic
        )

def get_mean_dic_3(dic_1, dic_2, dic_3):
    dic_characteristics = {
        'subtlety': (dic_1['subtlety'] + dic_2['subtlety'] + dic_3['subtlety']) / 3,
        'internalStructure': (dic_1['internalStructure'] + dic_2['internalStructure'] + dic_3['internalStructure']) / 3,
        'calcification': (dic_1['calcification'] + dic_2['calcification'] + dic_3['calcification']) / 3,
        'sphericity': (dic_1['sphericity'] + dic_2['sphericity'] + dic_3['sphericity']) / 3,
        'margin': (dic_1['margin'] + dic_2['margin'] + dic_3['margin']) / 3,
        'lobulation': (dic_1['lobulation'] + dic_2['lobulation'] + dic_3['lobulation']) / 3,
        'spiculation': (dic_1['spiculation'] + dic_2['spiculation'] + dic_3['spiculation']) / 3,
        'texture': (dic_1['texture'] + dic_2['texture'] + dic_3['texture']) / 3,
        'malignancy': (dic_1['malignancy'] + dic_2['malignancy'] + dic_3['malignancy']) / 3,
    }
    return dic_characteristics

def get_mean_dic_4(dic_1, dic_2, dic_3, dic_4):
    dic_characteristics = {
        'subtlety': (dic_1['subtlety'] + dic_2['subtlety'] + dic_3['subtlety'] + dic_3['subtlety']) / 4,
        'internalStructure': (dic_1['internalStructure'] + dic_2['internalStructure'] + dic_3['internalStructure'] + dic_3['internalStructure']) / 4,
        'calcification': (dic_1['calcification'] + dic_2['calcification'] + dic_3['calcification'] + dic_3['calcification']) / 4,
        'sphericity': (dic_1['sphericity'] + dic_2['sphericity'] + dic_3['sphericity'] + dic_3['sphericity']) / 4,
        'margin': (dic_1['margin'] + dic_2['margin'] + dic_3['margin'] + dic_3['margin']) / 4,
        'lobulation': (dic_1['lobulation'] + dic_2['lobulation'] + dic_3['lobulation'] + dic_3['lobulation']) / 4,
        'spiculation': (dic_1['spiculation'] + dic_2['spiculation'] + dic_3['spiculation'] + dic_3['spiculation']) / 4,
        'texture': (dic_1['texture'] + dic_2['texture'] + dic_3['texture'] + dic_3['texture']) / 4,
        'malignancy': (dic_1['malignancy'] + dic_2['malignancy'] + dic_3['malignancy'] + dic_3['malignancy']) / 4,
    }
    return dic_characteristics

def get_dic_characteristics(nodule):
    dic_characteristics = {
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
    return dic_characteristics
