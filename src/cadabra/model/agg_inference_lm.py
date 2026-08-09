import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pathlib
import os
from typing import List, Dict, Any
from copy import deepcopy
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

load_dotenv()

from cadabra.utils import read_json, write_json, generate_datetime_id
from cadabra.model.inference_lm import inference_lm
from cadabra.model.modeling_utils import get_model_name

def aggregate_inference_results(input_paths: List[str]) -> List:
    output_data = []

    for input_path in tqdm(input_paths, desc="Aggregating data"):
        input_data = read_json(input_path)
        metadata = input_data["metadata"]
        for result in input_data["data"]:
            output_data.append(
                {
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "metrics"
                    },
                    "gen_args": metadata["config"]["gen_args"],
                }
            )

    return output_data

def agg_inference_lm(config: DictConfig) -> str:
    """
    Runs inference with multiple generation argument configurations and aggregates results.
    """
    gen_args_lst = config.get("gen_args_lst", {})
    
    if not gen_args_lst:
        raise ValueError("gen_args_lst must be provided in config")
    
    print(f"Running inference with {len(gen_args_lst)} generation arg configurations")
    
    output_paths = []
    
    # Run inference for each generation args configuration
    for _, gen_args in gen_args_lst.items():
        print(f"Running inference with gen_args: {gen_args}")
        
        # Create a modified config with the specific gen_args
        run_config = deepcopy(config)
        run_config.gen_args = gen_args
    
        # Run inference
        paths = inference_lm(run_config)
        output_paths.extend(paths)
    
    # Aggregate results if configured
    print("Aggregating results from all inference runs")
    aggregated_results = aggregate_inference_results(output_paths)

    run_id = generate_datetime_id()
    output_dir = os.path.join(config.output_dir, "aggregated", run_id)
    model_name = get_model_name(config.model_path)
    os.makedirs(output_dir, exist_ok=True)
    agg_output_path = os.path.join(
            output_dir, f"{pathlib.Path(config.data_path).stem}_{model_name}_{run_id}_agg.json"
        )
    agg_output = {
        "metadata": {
            "agg_output_path": str(agg_output_path),
            "output_paths": output_paths,
            "size": len(aggregated_results),
            "model_name": model_name,
            "output_dir": output_dir,
            "run_id": run_id,
            "config": OmegaConf.to_container(config, resolve=True),
        },
        "metrics": {},
        "data": aggregated_results,
    }
    write_json(agg_output, agg_output_path)
    print(f"Aggregated results available at: {agg_output_path}")
    return agg_output_path


@hydra.main(version_base=None, config_path="configs", config_name="agg_inference_lm")
def main(config: DictConfig):
    agg_inference_lm(config)

if __name__ == "__main__":
    main()