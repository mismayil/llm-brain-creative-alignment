#!/bin/bash

python -m cadabra.alignment.generate_alignment_scripts \
    alignment_method="rsa_per_subject" \
    brains=["high_cre","low_cre"] \
    output_path="experiments/templeton_aut/aut_rsa_per_subject_high_low/run_rsa_per_subject_alignments.sh"