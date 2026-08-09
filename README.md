# Large Language Models Align with the Human Brain during Creative Thinking

![framework figure](./pipeline.png)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.03480)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

## Setup
First, install the dependencies:
```sh
pip install -r requirements.txt
```

Then install the package:
```sh
pip install -e .
```
This project uses [Hydra](https://hydra.cc/) to manage configurations which allows composing and importing config files.

## Alignment pipeline
At a high level, the pipeline for computing alignments can look like below:

[Raw brain data] --> [extract beta maps](#) --> [prepare brain data](#), [compute noise ceiling](#) --> [Brain activations, Noise ceilings]

[Raw task data] --> [prepare task data](#) --> [prepare lm eval data](#) --> [model inference](#model-inference) --> [extract model activations](#extracting-model-activations) --> [Model activations]

[Brain activations, Noise ceilings, Model activations] --> [compute alignments](#computing-alignments)

## Preparing brain data

### Extracting beta maps
Beta maps can be extracted using the First-level GLM analysis.
```sh
./experiments/templeton_aut/run_first_level_glm.sh
```

### Computing noise ceilings
Noise ceilings are computed on the beta maps.
```sh
./experiments/templeton_aut/compute_noise_ceiling.sh
```

## Preparing model data
Model scripts live under [src/cadabra/model](src/cadabra/model) and corresponding configs are in [src/cadabra/model/configs](src/cadabra/model/configs). 

### Preparing model task data
```sh
./experiments/templeton_aut/prepare_task_data.sh
```

### Preparing LM eval data
```sh
./experiments/templeton_aut/prepare_task_data.sh
```

### Model inference
Model inference can be done using the [`inference_lm.py`](src/cadabra/model/inference_lm.py) script. Here is an example:
```sh
python -m cadabra.model.inference_lm \
    model_path="meta-llama/Llama-3.1-8B-Instruct" \
    data_path="experiments/templeton_aut/data/eval/templeton_aut_create_eval_data.json" \
    output_dir="/path/to/inference/output/directory" \
    gen_args="sampling_t0.7_p0.95" \
    gen_args.max_new_tokens=1024
```
Note that it is better to log outputs to outside the git repo as they are big. Also note that since we use Hydra, we can reference entire config files on CLI like we did above for `gen_args="sampling_t0.7_p0.95"`, this refers to the sampling parameters specified in [sampling_t0.7_p0.95.yaml](src/cadabra/model/configs/gen_args/sampling_t0.7_p0.95.yaml) which itself imports from base [sampling.yaml](src/cadabra/model/configs/gen_args/sampling.yaml).
Eval data for model inference lives in `experiments` folder under the corresponding dataset directory (e.g. `experiments/templeton_aut/data/eval`). Check out [inference_lm.sh](experiments/templeton_aut/inference_lm.sh) for more examples.

### Extracting model activations
In order to get model activations, typically you would run model inference first (see [Model inference](#model-inference)) and then run the activation extraction on the results, but this script can be run on any datapath that has the appropriate fields.
Model activations can be extracted using the [extract_lm_activations.py](src/cadabra/model/extract_lm_activations.py). Here is an example:
```sh
python -m cadabra.model.extract_lm_activations \
       data_path="/path/to/inference/output/directory"
       prompt_only=True
```
The script by default will use the `metadata.config.model_path` to load the model and the tokenizer, however, this value can be overridden with the `model_path` option. Additionally, by default, the script will attempt to extract activations for `user_prompt` + `output` (concatenation is done appropriately using the chat template if it exists, otherwise defaults to simple string concat), so it will expect the `output` field. If you want to extract the activations only from the prompt, then pass `prompt_only=True` option. Activations by default will be saved under the the same model inference output directory and will be printed out. Check out [extract_lm_activations.sh](experiments/templeton_aut/extract_lm_activations.sh) for more examples.

## Computing alignments
Brain alignment config files can be found [here](src/cadabra/alignment/configs). [Default config](src/cadabra/alignment/configs/brain_alignment.yaml) contains reasonable default values and can be modified in CLI easily. Alignment can be computed using the [brain_alignment.py](src/cadabra/alignment/brain_alignment.py) script. Here is an example of a single alignment run:
```sh
python -m cadabra.alignment.brain_alignment \
    alignment="ridge" \
    model_args.model_datapath="/path/to/model/activations/data/json" \
    brain_args.brain_datapath="experiments/templeton_aut/data/brain/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json" \
    brain_args.noise_ceiling_path="experiments/templeton_aut/data/brain/yeo_dmn_brain_aut_beta_map_data/noise_ceilings/noise_ceiling_per_voxel_n10_20260117_224707.json" \
    model_args.model_data_sampling="time:-1::layer:1:"
``` 

This command computes the ridge regression alignment between the last-token (-1) activations of all layers of the model (except the first layer which is the embedding layer) and the beta maps of the DMN region of the brain and reports both raw and noise ceiling adjusted results. The command also saves the results to a local path and reports to wandb (for wandb, make sure to configure wandb API key by either exporting it or setting in the top-level .env file). 

Multiple alignment runs can be easily done by specifying `--multi-run` option and setting multiple setting values using comma like below:
```sh
python -m cadabra.alignment.brain_alignment \
    alignment="ridge" \
    model_args.model_datapath="path/to/model/data1","path/to/model/data2" \
    brain_args.brain_datapath="experiments/templeton_aut/data/brain/yeo_dmn_brain_aut_beta_map_data/first_level_glm_results_20260117_204028_dt_create.json" \
    brain_args.noise_ceiling_path="experiments/templeton_aut/data/brain/yeo_dmn_brain_aut_beta_map_data/noise_ceilings/noise_ceiling_per_voxel_n10_20260117_224707.json" \
    model_args.model_data_sampling="time:-1::layer:1:","time:mean::layer:1:"
```
Here we have specified multiple model data files and multiple model data sampling options (last-token and mean-token). Check out [aut_brain_alignment.sh](experiments/templeton_aut/aut_brain_alignment.sh) for more examples.