path_list_3_2 = '/Users/shirui/study/tianjin-university-study/application-of-deep-learning-in-CT-image-diagnosis-of-pulmonary-nodules/data/dataset_lidc/list3.2.csv'
root_lidc = '/Volumes/shirui_WD_2/lung_image/LIDC-IDRI'

import os
import glob

recursive_path_xml = os.path.join(root_lidc, '**', '*.xml')
list_path_xml = glob.glob(recursive_path_xml, recursive=True)

for each_path_xml in list_path_xml:
    lidc_no = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(each_path_xml))))
    lidc_sub = os.path.basename(os.path.dirname(os.path.dirname(each_path_xml)))
    lidc_sub_sub = os.path.basename(os.path.dirname(each_path_xml))
    name_xml = os.path.basename(each_path_xml)

# code get_annotation.py