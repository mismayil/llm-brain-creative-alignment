#!/bin/bash

DATA_DIR=$(realpath ../data/Templeton_fMRI_data)
python -m cadabra.brain.clean_fmri_prep_data -d "$DATA_DIR/templeton_aut_raw"