#!/bin/bash

DATA_DIR=$(realpath ../data/Templeton_fMRI_data)

# ################### Brain data from first level GLM results (OCT) ###########################
# FLM_RESULTS_PATH=$DATA_DIR/first_level_glm/20260326_102919/first_level_glm_results_20260326_102919.json

# # OCT Default Mode Network (DMN) data with NA responses removed
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_dmn_brain_oct_beta_map_data \
#         --roi yeo:dmn \
#         --roi-resample \
#         --roi-interpolation nearest \
#         --condition object \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_object_beta_map" \
#         --remove-na-responses

# # OCT Frontoparietal (FP) data with NA responses removed
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_fp_brain_oct_beta_map_data \
#         --roi yeo:fp \
#         --roi-resample \
#         --roi-interpolation nearest \
#         --condition object \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_object_beta_map" \
#         --remove-na-responses

################### Stimuli rating-based filtered brain data from first level GLM results ###########################
# AUT Default Mode Network (DMN) data with ratings
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#         -p add_rating \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_dmn_brain_aut_beta_map_data \
#         --skip-dim-mismatch-filter \
#         --remove-na-responses

thresholds=(
        1.75 
        # 2.0 
        2.25
)

for threshold in "${thresholds[@]}"; do
    # AUT Default Mode Network (DMN) data with ratings >= threshold
    python -m cadabra.brain.prepare_brain_data \
            -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
            -p post_filter \
            -m $DATA_DIR/mri_responses.xlsx \
            -o $DATA_DIR/yeo_dmn_brain_aut_beta_map_data \
            --min-stimuli-rating $threshold \
            --skip-dim-mismatch-filter \
            --remove-na-responses \
            --suffix "_rating_ge_${threshold}"

    # AUT Default Mode Network (DMN) data with ratings < threshold
    python -m cadabra.brain.prepare_brain_data \
            -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
            -p post_filter \
            -m $DATA_DIR/mri_responses.xlsx \
            -o $DATA_DIR/yeo_dmn_brain_aut_beta_map_data \
            --max-stimuli-rating $threshold \
            --skip-dim-mismatch-filter \
            --remove-na-responses \
            --suffix "_rating_lt_${threshold}"
done

# AUT Default Mode Network (DMN) data subject rating >= 2.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#         -p add_rating \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_dmn_brain_aut_beta_map_data \
#         --min-subject-rating 2.0 \
#         --skip-dim-mismatch-filter \
#         --remove-na-responses

# # AUT Frontoparietal (FP) data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#         -p add_rating \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_fp_brain_aut_beta_map_data \
#         --skip-dim-mismatch-filter \
#         --remove-na-responses

# # # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 3.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#         -p add_rating \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data \
#         --skip-dim-mismatch-filter \
#         --remove-na-responses

# # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 6.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#         -p add_rating \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data \
#         --skip-dim-mismatch-filter \
#         --remove-na-responses

# # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 9.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#         -p add_rating \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data \
#         --skip-dim-mismatch-filter \
#         --remove-na-responses

################### Brain data from first level GLM results ###########################
# FLM_RESULTS_PATH=$DATA_DIR/first_level_glm/20260117_204028/first_level_glm_results_20260117_204028.json

# # # AUT Default Mode Network (DMN) data with NA responses removed
# # python -m cadabra.brain.prepare_brain_data \
# #         -d $FLM_RESULTS_PATH \
# #         -p templeton_aut_flm \
# #         -m $DATA_DIR/mri_responses.xlsx \
# #         -o $DATA_DIR/yeo_dmn_brain_aut_beta_map_data \
# #         --roi yeo:dmn \
# #         --roi-resample \
# #         --roi-interpolation nearest \
# #         --condition create \
# #         --exclude-subjects 2000 \
# #         --sub-stimuli-field "contrast_create_beta_map" \
# #         --remove-na-responses

# # AUT Frontoparietal (FP) data with NA responses removed
# # python -m cadabra.brain.prepare_brain_data \
# #         -d $FLM_RESULTS_PATH \
# #         -p templeton_aut_flm \
# #         -m $DATA_DIR/mri_responses.xlsx \
# #         -o $DATA_DIR/yeo_fp_brain_aut_beta_map_data \
# #         --roi yeo:fp \
# #         --roi-resample \
# #         --roi-interpolation nearest \
# #         --condition create \
# #         --exclude-subjects 2000 \
# #         --sub-stimuli-field "contrast_create_beta_map" \
# #         --remove-na-responses

# AUT Somatomotor (SOM) data with NA responses removed
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_som_brain_aut_beta_map_data \
#         --roi yeo:som \
#         --roi-resample \
#         --roi-interpolation nearest \
#         --condition create \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_create_beta_map" \
#         --remove-na-responses

# # # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 3.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 3.0 \
#         --condition create \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_create_beta_map"

# # # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 3.0 with NA responses removed
# # python -m cadabra.brain.prepare_brain_data \
# #         -d $FLM_RESULTS_PATH \
# #         -p templeton_aut_flm \
# #         -m $DATA_DIR/mri_responses.xlsx \
# #         -o $DATA_DIR/no_na_resp/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data \
# #         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
# #         --roi-threshold 3.0 \
# #         --condition create \
# #         --exclude-subjects 2000 \
# #         --sub-stimuli-field "contrast_create_beta_map" \
# #         --remove-na-responses

# # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 6.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 6.0 \
#         --condition create \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_create_beta_map"

# # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 6.0 with NA responses removed
# # python -m cadabra.brain.prepare_brain_data \
# #         -d $FLM_RESULTS_PATH \
# #         -p templeton_aut_flm \
# #         -m $DATA_DIR/mri_responses.xlsx \
# #         -o $DATA_DIR/no_na_resp/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data \
# #         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
# #         --roi-threshold 6.0 \
# #         --condition create \
# #         --exclude-subjects 2000 \
# #         --sub-stimuli-field "contrast_create_beta_map" \
# #         --remove-na-responses

# # # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 9.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 9.0 \
#         --condition create \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_create_beta_map"

# AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 9.0 with NA responses removed
# python -m cadabra.brain.prepare_brain_data \
#         -d $FLM_RESULTS_PATH \
#         -p templeton_aut_flm \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/no_na_resp/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 9.0 \
#         --condition create \
#         --exclude-subjects 2000 \
#         --sub-stimuli-field "contrast_create_beta_map" \
#         --remove-na-responses

################### Brain data from raw fMRI data ###########################
# AUT whole brain data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/brain_aut_data \
#         --condition create \
#         --exclude-subjects 2000

# AUT Default Mode Network (DMN) data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_dmn_brain_aut_data \
#         --roi yeo:dmn \
#         --roi-resample \
#         --roi-interpolation nearest \
#         --condition create \
#         --exclude-subjects 2000

# AUT Frontoparietal (FP) data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_fp_brain_aut_data \
#         --roi yeo:fp \
#         --roi-resample \
#         --roi-interpolation nearest \
#         --condition create \
#         --exclude-subjects 2000

# # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 3.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 3.0 \
#         --condition create \
#         --exclude-subjects 2000

# # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 6.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 6.0 \
#         --condition create \
#         --exclude-subjects 2000

# # AUT masked with Templeton AUT-OCT SLM z_score atlas data with threshold 9.0
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_data \
#         --roi-path $DATA_DIR/second_level_glm/20260116_110829/second_level_create_vs_object_z_map.nii.gz \
#         --roi-threshold 9.0 \
#         --condition create \
#         --exclude-subjects 2000

# OCT whole brain data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/brain_oct_data \
#         --condition object

# OCT Default Mode Network (DMN) data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/yeo_dmn_brain_oct_data \
#         --roi yeo:dmn \
#         --roi-resample \
#         --roi-interpolation nearest \
#         --condition object

# AUT masked with Templeton AUT-OCT atlas data
# python -m cadabra.brain.prepare_brain_data \
#         -d $DATA_DIR/templeton_aut_raw \
#         -m $DATA_DIR/mri_responses.xlsx \
#         -o $DATA_DIR/templeton_aut_sub_oct_brain_aut_data \
#         --roi-path $DATA_DIR/templeton_aut_sub_oct_atlas.nii.gz \
#         --condition create