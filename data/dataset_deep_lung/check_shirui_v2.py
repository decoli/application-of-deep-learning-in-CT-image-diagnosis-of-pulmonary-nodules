import pandas as pd

path_shirui_v2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2.csv'

pd_shirui_v2 = pd.read_csv(path_shirui_v2)
count = 0
for index, each_annotation in pd_shirui_v2.iterrows():
    condition_1 = each_annotation['malignant'] == 1 and each_annotation['malignancy'] < 3
    condition_2 = each_annotation['malignant'] == 0 and each_annotation['malignancy'] > 3
    if condition_1 or condition_2:
        count += 1
        print(each_annotation)
print(count)