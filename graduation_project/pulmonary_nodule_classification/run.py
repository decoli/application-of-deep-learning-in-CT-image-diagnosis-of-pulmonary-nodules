import os

for time in range(10):
    for use_cross in range(1, 5):
        run_command = 'graduation_project/pulmonary_nodule_classification/classification_model_of_prior_knowledge_advanced.py --use-cross {use_cross}'.format(use_cross=use_cross + 1)
        os.system(run_command)
