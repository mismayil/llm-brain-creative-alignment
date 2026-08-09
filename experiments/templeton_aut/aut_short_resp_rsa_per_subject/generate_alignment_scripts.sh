#!/bin/bash

python -m cadabra.alignment.generate_alignment_scripts \
    alignment_method="rsa_per_subject" \
    activation_modes=["short_prompt","short_model_resp"] \
    output_path="experiments/templeton_aut/aut_short_resp_rsa_per_subject/run_rsa_per_subject_alignments.sh"