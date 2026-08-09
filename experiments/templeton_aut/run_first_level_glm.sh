#!/bin/bash

DATA_DIR=$(realpath ../data/Templeton_fMRI_data)

# treatment vs control contrast
# python -m cadabra.brain.run_first_level_glm 
    # -d "$DATA_DIR"/templeton_aut_raw \
    # -o "$DATA_DIR/first_level_glm" \
    # --exclude-subjects 2000 \
    # --parallel \
    # --num-workers 4 \
    # --contrast-def "treatment-control"

# create beta map per stimulus
# python -m cadabra.brain.run_first_level_glm \
#     -d "$DATA_DIR"/templeton_aut_raw \
#     -o "$DATA_DIR/first_level_glm" \
#     --exclude-subjects 2000 \
#     --parallel \
#     --num-workers 4 \
#     --contrast-def "treatment" \
#     --compute-per-stimulus \
#     --treatment-task "create"

# object beta map per stimulus
python -m cadabra.brain.run_first_level_glm \
    -d "$DATA_DIR"/templeton_aut_raw \
    -o "$DATA_DIR/first_level_glm" \
    --exclude-subjects 2000 \
    --parallel \
    --num-workers 4 \
    --contrast-def "treatment" \
    --compute-per-stimulus \
    --treatment-task "object"