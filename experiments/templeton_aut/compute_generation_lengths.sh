#!/bin/bash

python experiments/templeton_aut/compute_generation_length_stats.py \
    --yaml-path src/cadabra/alignment/configs/alignment_data/aut.yaml \
    --datapath-name model_resp \
    --model-dir ../outputs/templeton_aut