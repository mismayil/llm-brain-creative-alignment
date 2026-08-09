#!/bin/bash

python -m cadabra.alignment.generate_alignment_scripts \
    alignment_method="rsa_per_subject" \
    output_path="experiments/templeton_aut/aut_rsa_per_subject/run_rsa_per_subject_alignments.sh"

python -m cadabra.alignment.generate_alignment_scripts \
    alignment_method="rsa_per_subject" \
    brains=["yeo_dmn"] \
    activation_modes=["empty_prompt","empty_model_resp","nolang_prompt","nolang_model_resp"] \
    output_path="experiments/templeton_aut/aut_rsa_per_subject/run_rsa_per_subject_alignments2.sh"