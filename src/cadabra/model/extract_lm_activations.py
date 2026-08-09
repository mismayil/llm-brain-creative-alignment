import argparse
import numpy as np
import pathlib
import os
from tqdm import tqdm
import math
import hydra
from omegaconf import DictConfig, OmegaConf

from cadabra.utils import generate_datetime_id, read_json, write_json, batched, find_files
from cadabra.model.modeling_utils import load_lm, get_model_name, prepare_lm_inputs, extract_model_activations

def _save_activations_for_sample(sample, activations, output_dir):
    activations_filepath = f"{sample['id']}_{sample['result_id']}_activations.npy"
    activations_dir = "activations"
    activations_path = os.path.join(activations_dir, activations_filepath)
    full_activations_path = os.path.join(output_dir, activations_path)
    os.makedirs(os.path.join(output_dir, activations_dir), exist_ok=True)
    np.save(full_activations_path, activations)
    return activations_path

def extract_batch_activations(
    model,
    tokenizer,
    batch, 
    prompt_only: bool = False
):
    """
    Extract model activations for a batch of samples.
    Arguments:
        model: the language model
        tokenizer: the tokenizer
        batch: list of samples, each with "user_prompt" and optionally "output"
        prompt_only: whether to extract activations for prompt tokens only
    Returns:
        batch_activations: (B, S, L, D) array of activations.
    """
    add_responses = not prompt_only and batch[0].get("output", None) is not None
    inputs = prepare_lm_inputs(tokenizer, prompts=[sample["user_prompt"] for sample in batch], 
                               responses=[sample.get("output", None) for sample in batch] if add_responses else None)
    batch_activations = extract_model_activations(model, inputs)
    return batch_activations  # (B, S, L, D)

def extract_activations_for_datapath(config, datapath):
    """
    Extract activations for all samples in a data file.
    Arguments:
        config: configuration object
        datapath: path to the data file
    Returns:
        output_dir: directory where activations are saved
    """
    datapath = pathlib.Path(datapath)
    input_data = read_json(datapath)
    metadata = input_data.get("metadata", {})
    data = input_data["data"]

    if config.model_path is not None:
        model_path = config.model_path
    else:
        model_path = metadata["config"]["model_path"]
    
    print(f"Loading model and tokenizer from {model_path}")
    model, tokenizer = load_lm(model_path, from_pretrained=not config.model_from_untrained, padding_side="left")

    if config.num_samples and config.num_samples > 0:
        data = data[: int(config.num_samples)]

    model_name = get_model_name(model_path)
    run_id = generate_datetime_id()
    output_dir = datapath.parent / "activations" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{datapath.stem}_with_activations.json"

    print(f"Writing to {output_path}")

    outputs = {
        "metadata": {
            "output_path": str(output_path),
            "size": len(data),
            "model_path": model_path,
            "model_name": model_name,
            "output_dir": str(output_dir),
            "run_id": run_id,
            "config": OmegaConf.to_container(config, resolve=True),
        },
        "metrics": {},
        "data": data,
    }

    for batch in tqdm(
        batched(data, size=config.batch_size),
        total=math.ceil(len(data) / config.batch_size),
    ):
        filtered_batch = []

        for sample in batch:
            if "activations_path" in sample:
                continue

            filtered_batch.append(sample)

        model_activations = extract_batch_activations(
            model,
            tokenizer,
            filtered_batch,
            prompt_only=config.prompt_only
        )

        for sample, activations in zip(filtered_batch, model_activations):
            activations_path = _save_activations_for_sample(sample, activations, output_dir)
            sample["activations_path"] = activations_path
        
        write_json(outputs, output_path)

    print("Done.")

    return output_dir

def extract_activations(config: DictConfig):
    datapaths = []

    if pathlib.Path(config.data_path).is_file():
        datapaths.append(config.data_path)
    else:
        datapaths.extend(find_files(config.data_path, "json"))

    for datapath in datapaths:
        extract_activations_for_datapath(config, datapath)   

    print(f"All done. Activations extracted for {len(datapaths)} data files.")

@hydra.main(version_base=None, config_path="configs", config_name="extract_lm_activations")
def main(config: DictConfig):
    extract_activations(config) 

if __name__ == "__main__":
    main()