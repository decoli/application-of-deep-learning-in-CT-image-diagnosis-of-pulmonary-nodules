path_list_3_2 = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/list3.2.csv'
root_lidc = '/Volumes/shirui_WD_2/lung_image/LIDC-IDRI'
test_root_lidc = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/image/LIDC-IDRI'

import os
from bs4 import BeautifulSoup
import glob
import pandas as pd
import pydicom

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

    # parsing diagnostic information
    reading_sessions = xml.LidcReadMessage.find_all('readingSession')