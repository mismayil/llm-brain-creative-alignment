#!/bin/bash

# This script uses the Globus CLI to download specific fMRI files from a Globus endpoint.
# Make sure you have the Globus CLI installed and authenticated before running this script.
# You also need to have a Globus Connect Personal endpoint set up for your local machine.
# First download the Globus CLI and authenticate:
# https://docs.globus.org/cli/
# Then setup your Globus Connect Personal endpoint:
# https://docs.globus.org/globus-connect-personal/

# Set your source and destination endpoints and paths
SRC_ENDPOINT="6d9cc7bd-75d2-49ff-884c-a7dde4c16e30"
SRC_PATH="/"   # e.g. /your/data/folder/
DEST_ENDPOINT="4cb2cab4-4afc-11f0-9425-02fa2a4031ab" # For your local machine, use your Globus Connect Personal endpoint UUID
DEST_PATH="/Users/mismayil/Desktop/phd/projects/project-cadabra/data/Templeton_fMRI_data/"     # e.g. /Users/yourname/data/
TASK="dt"
SPACE="MNI152NLin2009cAsym"
SUBS_WITH_NO_BOLD_GZ=(2006 2007 2008 2010 2011 2012 2013 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2027 2030 2032 2034 2035 2037 2038 2039 2040 2044 2046 2047 2048 2076 2091 2139 2146)

# Patterns to match
PATTERNS=(
    # "^sub-[0-9]+_task-${TASK}_space-${SPACE}_desc-preproc_bold.nii.gz$"
    # "^sub-[0-9]+_task-${TASK}_space-${SPACE}_desc-brain_mask.nii.gz$"
    # "^sub-[0-9]+_task-${TASK}_desc-confounds_regressors.tsv$"
    "^sub-[0-9]+_task-${TASK}_space-${SPACE}_desc-preproc_bold.nii$"
    # "^sub-[0-9]+_task-${TASK}_run-01_events.tsv$"
)

# List all subject/func files matching patterns
for subdir in $(globus ls ${SRC_ENDPOINT}:${SRC_PATH}); do
    sub_id=$(echo ${subdir} | grep -oE '\d+')
    # echo "Processing subject ID: $sub_id"
    # proceed if sub_id is greater than or equal to 2036
    # if [[ -z "$sub_id" || "$sub_id" -lt 2036 ]]; then
    #     continue
    # fi

    # skip if sub_id is not in the list of subjects with no bold.nii.gz files
    # if [[ " ${SUBS_WITH_NO_BOLD_GZ[*]} " != *" $sub_id "* ]]; then
    #     # echo "Skipping subject ID: $sub_id (has bold.nii.gz files)"
    #     continue
    # fi
    echo "Downloading files for subject ID: $sub_id"
    funcdir="${SRC_PATH}${subdir}func/"
    # Check if func directory exists
    if globus ls ${SRC_ENDPOINT}:${funcdir} &>/dev/null; then
        for pattern in "${PATTERNS[@]}"; do
            files=$(globus ls ${SRC_ENDPOINT}:${funcdir} | grep -E "${pattern}")
            for file in $files; do
                src_file="${funcdir}${file}"
                dest_dir="${DEST_PATH}${subdir}func/"
                mkdir -p "${dest_dir}"
                echo "Transferring ${src_file} to ${dest_dir}${file}"
                # Transfer the file using Globus CLI
                # Use --notify off to suppress notifications
                globus transfer --notify off ${SRC_ENDPOINT}:${src_file} ${DEST_ENDPOINT}:${dest_dir}${file}
            done
        done
    fi
done