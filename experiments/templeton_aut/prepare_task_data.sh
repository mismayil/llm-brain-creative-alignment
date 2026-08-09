#!/bin/bash

# AUT without subject IDs
python -m cadabra.prepare_task_data \
        -i ../data/Templeton_fMRI_data/mri_responses.xlsx \
        -o experiments/templeton_aut/data/task/templeton_aut_create_task_data.json \
        -t templeton_aut -- --condition "create"

# AUT with subject IDs
# python -m cadabra.prepare_task_data \
#         -i ../data/Templeton_fMRI_data/mri_responses.xlsx \
#         -o experiments/templeton_aut/data/task/templeton_aut_create_task_data_with_subjects.json \
#         -t templeton_aut_with_subjects -- --condition "create"

# # OCT without subject IDs
python -m cadabra.prepare_task_data \
        -i ../data/Templeton_fMRI_data/mri_responses.xlsx \
        -o experiments/templeton_aut/data/task/templeton_aut_object_task_data.json \
        -t templeton_aut -- --condition "object"

# # OCT with subject IDs
# python -m cadabra.prepare_task_data \
#         -i ../data/Templeton_fMRI_data/mri_responses.xlsx \
#         -o experiments/templeton_aut/data/task/templeton_aut_object_task_data_with_subjects.json \
#         -t templeton_aut_with_subjects -- --condition "object"