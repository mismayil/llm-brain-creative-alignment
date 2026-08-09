#!/bin/bash

python -m cadabra.alignment.generate_alignment_scripts \
    alignment_method="rsa_per_subject" \
    alignment_data="oct" \
    brains=["yeo_dmn","yeo_fp"] \
    output_path="experiments/templeton_aut/oct_rsa_per_subject/run_rsa_per_subject_alignments.sh"