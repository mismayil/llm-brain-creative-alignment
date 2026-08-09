#!/bin/bash

DATA_DIR=$(realpath ../data/Templeton_fMRI_data)
FLM_RESULTS_PATH=$(realpath ../data/Templeton_fMRI_data/first_level_glm/20260116_104245/first_level_glm_results_20260116_104245.json)
python -m cadabra.brain.run_second_level_glm -i "$FLM_RESULTS_PATH" -o "$DATA_DIR/second_level_glm"