#!/bin/bash

# AUT (create) task
# DATA_DIR=$(realpath ../data/Templeton_fMRI_data)
# python -m cadabra.brain.extract_fmri_task_data -d $DATA_DIR/templeton_aut_raw --condition "create"

# OCT (object) task
DATA_DIR=$(realpath ../data/Templeton_fMRI_data)
python -m cadabra.brain.extract_fmri_task_data -d $DATA_DIR/templeton_aut_raw --condition "object"