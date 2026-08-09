#!/bin/bash

DATA_DIR=$(realpath ../data/Templeton_fMRI_data)

################# Compute RSA per subject noise ceiling OCT beta map brain data ###########################
# # Default Mode Network (DMN) region
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_dmn_brain_oct_beta_map_data/first_level_glm_results_20260326_102919_dt_object.json \
#         --type "rsa_per_subject"

# # Default Mode Network (DMN) region
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_fp_brain_oct_beta_map_data/first_level_glm_results_20260326_102919_dt_object.json \
#         --type "rsa_per_subject"

################# Compute RSA per subject noise ceiling for stimuli rating-filtered beta map brain data ###########################
thresholds=(
        1.75 
        # 2.0 
        2.25
)

for threshold in "${thresholds[@]}"; do
    # AUT Default Mode Network (DMN) data with ratings >= threshold
    python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings_post_filtered_rating_ge_${threshold}.json \
            --type "rsa_per_subject"

    # AUT Default Mode Network (DMN) data with ratings < threshold
    python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings_post_filtered_rating_lt_${threshold}.json \
            --type "rsa_per_subject"
done

################# Compute mean voxel per subject noise ceiling for beta map brain data ###########################
# Default Mode Network (DMN) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel_per_subject"

# # Frontoparietal (FP) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel_per_subject"

# # # custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel_per_subject"

# # # custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel_per_subject"

# # # custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel_per_subject"

################# Compute RSA noise ceiling for beta map brain data ###########################
# Default Mode Network (DMN) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa"

# # Frontoparietal (FP) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa"

# # # custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa"

# # # custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa"

# # # custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa"

################# Compute RSA per subject noise ceiling for raw brain data ###########################
# Default Mode Network (DMN) region
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_dmn_brain_aut_data/templeton_aut_raw_dt_create.json --type "rsa_per_subject"

# # Frontoparietal (FP) region
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_fp_brain_aut_data/templeton_aut_raw_dt_create.json --type "rsa_per_subject"

# # custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_data/templeton_aut_raw_dt_create.json --type "rsa_per_subject"

# # custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_data/templeton_aut_raw_dt_create.json --type "rsa_per_subject"

# # custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_data/templeton_aut_raw_dt_create.json --type "rsa_per_subject"

################# Compute RSA per subject noise ceiling for beta map brain data ###########################
# Default Mode Network (DMN) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa_per_subject"

# Frontoparietal (FP) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa_per_subject"

# Somatomotor (SOM) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_som_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json \
#        --type "rsa_per_subject"

# # # custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa_per_subject"

# # # custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa_per_subject"

# # # custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "rsa_per_subject"

################# Compute mean voxel noise ceiling for beta map brain data ###########################
# # Default Mode Network (DMN) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel"

# # Frontoparietal (FP) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel"

# # # custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel"

# # # custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel"

# # # custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "mean_voxel"

################# Compute median voxel noise ceiling for beta map brain data ###########################
# Default Mode Network (DMN) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "median_voxel"

# # Frontoparietal (FP) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "median_voxel"

# # # custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "median_voxel"

# # # custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "median_voxel"

# # # custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling \
#        -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json \
#        --type "median_voxel"

################# Compute per voxel noise ceiling for beta map brain data ###########################
# # Default Mode Network (DMN) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json

# # Frontoparietal (FP) region beta map data from flm results
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_fp_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json

# custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json

# custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json

# custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create_with_ratings.json

################# Compute per voxel noise ceiling for raw brain data ###########################
# whole brain
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/brain_aut_data/templeton_aut_raw_dt_create.json

# Default Mode Network (DMN) region
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_dmn_brain_aut_data/templeton_aut_raw_dt_create.json

# Frontoparietal (FP) region
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/yeo_fp_brain_aut_data/templeton_aut_raw_dt_create.json

# custom creativity network, aut_sub_oct slm z_map th3
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th3_brain_aut_data/templeton_aut_raw_dt_create.json

# custom creativity network, aut_sub_oct slm z_map th6
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th6_brain_aut_data/templeton_aut_raw_dt_create.json

# custom creativity network, aut_sub_oct slm z_map th9
# python -m cadabra.brain.compute_noise_ceiling -d $DATA_DIR/templeton_aut_sub_oct_slm_z_map_th9_brain_aut_data/templeton_aut_raw_dt_create.json