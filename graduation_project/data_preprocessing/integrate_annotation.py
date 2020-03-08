path_list_3_2 = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/list3.2.csv'
root_lidc = '/Volumes/shirui_WD_2/lung_image/LIDC-IDRI'
test_root_lidc = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/image/LIDC-IDRI'
path_list_3_2_integrated = 'data/dataset_lidc/list3.2_integrated.csv'

root = root_lidc

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

recursive_path_xml = os.path.join(root, '**', '*.xml')
list_path_xml = glob.glob(recursive_path_xml, recursive=True)

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
        'subtlety': (dic_1['subtlety'] + dic_2['subtlety'] + dic_3['subtlety'] + dic_4['subtlety']) / 4,
        'internalStructure': (dic_1['internalStructure'] + dic_2['internalStructure'] + dic_3['internalStructure'] + dic_4['internalStructure']) / 4,
        'calcification': (dic_1['calcification'] + dic_2['calcification'] + dic_3['calcification'] + dic_4['calcification']) / 4,
        'sphericity': (dic_1['sphericity'] + dic_2['sphericity'] + dic_3['sphericity'] + dic_4['sphericity']) / 4,
        'margin': (dic_1['margin'] + dic_2['margin'] + dic_3['margin'] + dic_4['margin']) / 4,
        'lobulation': (dic_1['lobulation'] + dic_2['lobulation'] + dic_3['lobulation'] + dic_4['lobulation']) / 4,
        'spiculation': (dic_1['spiculation'] + dic_2['spiculation'] + dic_3['spiculation'] + dic_4['spiculation']) / 4,
        'texture': (dic_1['texture'] + dic_2['texture'] + dic_3['texture'] + dic_4['texture']) / 4,
        'malignancy': (dic_1['malignancy'] + dic_2['malignancy'] + dic_3['malignancy'] + dic_4['malignancy']) / 4,
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


with open(path_list_3_2_integrated, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(
        [
        'case',
        'scan',
        'roi',
        'volume',
        'eq. diam.',
        'x loc.',
        'y loc.',
        'slice no.',

        'subtlety',
        'internalStructure',
        'calcification',
        'sphericity',
        'margin',
        'lobulation',
        'spiculation',
        'texture',
        'malignancy',

        'class_malignant',
        'dir_dicom',
        ]
    )

for each_path_xml in list_path_xml:
    lidc_no = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(each_path_xml))))
    lidc_sub = os.path.basename(os.path.dirname(os.path.dirname(each_path_xml)))
    lidc_sub_sub = os.path.basename(os.path.dirname(each_path_xml))
    name_xml = os.path.basename(each_path_xml)

    # get dcm
    path_dicom = os.path.join(root, lidc_no, lidc_sub, lidc_sub_sub, '*.dcm')
    list_path_dicom = glob.glob(path_dicom)
    dicom = pydicom.read_file(list_path_dicom[0])
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
    for index_row, each_row in pd_scan.iterrows():
        print(each_row)

        #####
        try:
            math.isnan(each_row['Unnamed: 13'])
        except TypeError:
            None
        else:
            if not math.isnan(each_row['Unnamed: 13']): # 诊断太多， 大于医师人数
                continue
        
        try:
            math.isnan(each_row['Unnamed: 11'])
        except TypeError:
            None
        else:
            if math.isnan(each_row['Unnamed: 11']): # 诊断太少， 至少3个医师的诊断
                continue

        # get nodule IDs
        list_nodule_id = []
        list_nodule_id.append(each_row['nodIDs']) # 第1个诊断
        list_nodule_id.append(each_row['Unnamed: 10']) # 第2个诊断
        list_nodule_id.append(each_row['Unnamed: 11']) # 第3个诊断

        try:
            math.isnan(each_row['Unnamed: 12'])
        except TypeError:
            list_nodule_id.append(each_row['Unnamed: 12']) # 第4个诊断
        else:
            None

        # parsing diagnostic information
        flag_malignant_physician_1 = 0
        flag_malignant_physician_2 = 0
        flag_malignant_physician_3 = 0
        flag_malignant_physician_4 = 0

        flag_benign_physician_1 = 0
        flag_benign_physician_2 = 0
        flag_benign_physician_3 = 0
        flag_benign_physician_4 = 0

        list_dic_characteristics = []
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
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        elif diagnostic_index == 2:
                            flag_malignant_physician_2 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        elif diagnostic_index == 3:
                            flag_malignant_physician_3 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        elif diagnostic_index == 4:
                            flag_malignant_physician_4 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        break

                    elif malignancy < 3: # 判定为良性
                        if diagnostic_index == 1:
                            flag_benign_physician_1 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        elif diagnostic_index == 2:
                            flag_benign_physician_2 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        elif diagnostic_index == 3:
                            flag_benign_physician_3 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        elif diagnostic_index == 4:
                            flag_benign_physician_4 += 1
                            dic_characteristics = get_dic_characteristics(nodule)
                            list_dic_characteristics.append(dic_characteristics)
                        break

        count_malignant_physician = flag_malignant_physician_1 + flag_malignant_physician_2 + flag_malignant_physician_3 + flag_malignant_physician_4
        count_benign_physician = flag_benign_physician_1 + flag_benign_physician_2 + flag_benign_physician_3 + flag_benign_physician_4

        if count_malignant_physician >= 3:
            # 判定为恶性
            class_malignant = 1
        elif count_benign_physician >= 3:
            # 判定为良性
            class_malignant = 0
        else:
            continue # 医师对该结节没有诊断

        if ((count_malignant_physician == 3) or (count_benign_physician == 3)):
            mean_dic = get_mean_dic_3(list_dic_characteristics[0], list_dic_characteristics[1], list_dic_characteristics[2])
        elif ((count_malignant_physician == 4) or (count_benign_physician == 4)):
            mean_dic = get_mean_dic_4(list_dic_characteristics[0], list_dic_characteristics[1], list_dic_characteristics[2], list_dic_characteristics[3])

        # write info at new_annotation.csv 
        print(each_path_xml) # class_malignant, mean_dic

        write_row = []
        write_row.append(each_row['case'])
        write_row.append(each_row['scan'])
        write_row.append(each_row['roi'])
        write_row.append(each_row['volume'])
        write_row.append(each_row['eq. diam.'])
        write_row.append(each_row['x loc.'])
        write_row.append(each_row['y loc.'])
        write_row.append(each_row['slice no.'])

        write_row.append(mean_dic['subtlety'])
        write_row.append(mean_dic['internalStructure'])
        write_row.append(mean_dic['calcification'])
        write_row.append(mean_dic['sphericity'])
        write_row.append(mean_dic['margin'])
        write_row.append(mean_dic['lobulation'])
        write_row.append(mean_dic['spiculation'])
        write_row.append(mean_dic['texture'])
        write_row.append(mean_dic['malignancy'])

        write_row.append(class_malignant)
        write_row.append(os.path.join(lidc_no, lidc_sub, lidc_sub_sub))
        
        with open(path_list_3_2_integrated, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(write_row)