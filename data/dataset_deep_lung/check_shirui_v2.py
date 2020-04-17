import pandas as pd

path_shirui_v2 = 'data/dataset_deep_lung/annotationdetclssgm_doctor_shirui_v2_test.csv'

pd_shirui_v2 = pd.read_csv(path_shirui_v2)
count_1 = 0
count_2 = 0
for index, each_annotation in pd_shirui_v2.iterrows():
    condition_1 = each_annotation['malignant'] == 1 and each_annotation['malignancy'] < 3
    condition_2 = each_annotation['malignant'] == 0 and each_annotation['malignancy'] > 3
    if condition_1:
        count_1 += 1
    if condition_2:
        count_2 += 1

print(count_1)
print(count_2)