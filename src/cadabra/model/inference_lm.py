import argparse
from typing import Optional
import numpy as np
import pathlib
import os
from tqdm import tqdm
import math
import traceback
import dataclasses
from copy import deepcopy
import hydra
from omegaconf import DictConfig, OmegaConf

from cadabra.utils import read_json, write_json, generate_unique_id, batched, none_or_float, none_or_int, find_files, generate_datetime_id
from cadabra.model.modeling_utils import load_lm, get_model_name, prepare_lm_inputs, ablate_model, load_network_mask

@dataclasses.dataclass
class ModelOutput:
    text: Optional[str] = None
    usage: Optional[dict] = None
    exception: Optional[Exception] = None

def _write_error(error_path, sample, exception):
    with open(error_path, "a") as error_file:
        error_file.write(f"Error for sample {sample['id']}: {str(exception)}\n")
        error = "".join(
            traceback.format_exception(
                type(exception), value=exception, tb=exception.__traceback__
            )
        )
        error_file.write(error)
        error_file.write("\n")

def get_hf_gen_args(gen_args: dict) -> dict:
    hf_gen_args = {}
    valid_hf_gen_arg_names = ["temperature", "top_k", "top_p", "max_new_tokens", "do_sample", "typical_p", "min_p"]
    for arg_name, arg_value in gen_args.items():
        if arg_name in valid_hf_gen_arg_names:
            hf_gen_args[arg_name] = arg_value
    return hf_gen_args

def inference_hf_model(
    model,
    tokenizer,
    batch,
    gen_args: Optional[dict] = None
):
    if not gen_args:
        hf_gen_args = {}
    else:
        hf_gen_args = get_hf_gen_args(gen_args)

    inputs = prepare_lm_inputs(tokenizer, [sample["user_prompt"] for sample in batch])

    model_outputs = []

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        **hf_gen_args,
    )

    for i, output in enumerate(outputs):
        response = output[inputs["input_ids"].shape[-1] :]
        model_outputs.append(
            ModelOutput(
                text=tokenizer.decode(
                    response,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
        )

    return model_outputs

def inference_on_datapath(config, datapath, model, tokenizer, run_id, output_dir):
    datapath = pathlib.Path(datapath)
    input_data = read_json(datapath)
    data = input_data["data"]

    if not config.resume:
        if config.gen_args.num_samples and config.gen_args.num_samples > 0:
            data = data[: int(config.gen_args.num_samples)]
        inference_data = []
        for sample in data:
            for _ in range(config.gen_args.num_inferences):
                inference_data.append(deepcopy(sample))
        data = inference_data

    model_name = get_model_name(config.model_path)

    if config.resume:
        output_path = str(datapath)
        error_path = output_path.replace(".json", "_errors.txt")
    else:
        output_path = os.path.join(
            output_dir, f"{datapath.stem}_{model_name}_{run_id}.json"
        )
        error_path = os.path.join(
            output_dir, f"{datapath.stem}_{model_name}_{run_id}_errors.txt"
        )

    print(f"Writing to {output_path}")

    outputs = {
        "metadata": {
            **input_data["metadata"],
            "output_path": output_path,
            "size": len(data),
            "model_name": model_name,
            "output_dir": output_dir,
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
        try:
            filtered_batch = []

            for sample in batch:
                if "output" in sample:
                    continue

                filtered_batch.append(sample)

            if config.force_response:
                model_outputs = [ModelOutput(text=sample["response"]) for sample in filtered_batch]
            else:
                model_outputs = inference_hf_model(
                    model,
                    tokenizer,
                    filtered_batch,
                    gen_args=outputs["metadata"]["config"]["gen_args"],
                )

            for sample, result in zip(filtered_batch, model_outputs):
                sample["result_id"] = generate_unique_id()
                if result.text is not None:
                    sample["output"] = result.text
                if result.usage is not None:
                    sample["usage"] = result.usage
                if result.exception is not None:
                    sample["exception"] = str(result.exception)
            
            write_json(outputs, output_path)
        except Exception as e:
            _write_error(error_path, sample, e)
    
    print("Done.")

    return output_path

def inference_lm(config: DictConfig):
    np.random.seed(config.seed)

    run_id = generate_datetime_id()

    datapaths = []

    if pathlib.Path(config.data_path).is_file():
        datapaths.append(config.data_path)
    else:
        datapaths.extend(find_files(config.data_path, "json"))

    ablation_mask = None
    if config.ablation.path is not None:
        print(f"Loading ablation mask from {config.ablation.path}")
        ablation_mask = load_network_mask(config.ablation.path, 
                                           network_type=config.ablation.type, 
                                           ignore_first_layer=config.ablation.ignore_first_layer)
        print("Ablation mask loaded.")

    if config.force_response:
        print("Force response mode enabled. Model inference will be skipped.")
        model, tokenizer = None, None
    else:
        print(f"Loading model and tokenizer from {config.model_path}")
        model, tokenizer = load_lm(config.model_path, from_pretrained=not config.model_from_untrained)

    if ablation_mask is not None:
        print("Applying ablation to model.")
        ablate_model(model, ablation_mask)
    
    output_dir = os.path.join(config.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    for datapath in datapaths:
        output_path = inference_on_datapath(
            config,
            datapath,
            model=model,
            tokenizer=tokenizer,
            run_id=run_id,
            output_dir=output_dir
        )  
        output_paths.append(output_path)

    print(f"All done. Outputs are in {output_dir}")
    return output_paths

@hydra.main(version_base=None, config_path="configs", config_name="inference_lm")
def main(config: DictConfig):
    inference_lm(config)

if __name__ == "__main__":
    main()