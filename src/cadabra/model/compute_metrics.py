from dotenv import load_dotenv
import argparse
import pathlib
from tqdm import tqdm
import torch
import numpy as np
import asyncio

from cadabra.utils import find_files, read_json, write_json, compute_usage, batched
from cadabra.model.modeling_utils import load_classifier
from cadabra.model.inference_lm_api import get_inference_service
from cadabra.model.prompt_templates import LLM_AUT_SCORING_TEMPLATE

async def compute_usage_and_cost(input_data, source_data_dict=None, **kwargs):
    model = input_data["metadata"]["model_name"]
    data = input_data["data"]
    for sample in tqdm(data, desc="Computing usage and cost"):
        source_sample = source_data_dict.get(sample["id"], {}) if source_data_dict else {}
        usage, cost = compute_usage({**sample, **source_sample}, model=model)
        sample["usage"] = usage
        sample["cost"] = cost
    return {
        "usage": {
            "input_tokens": sum([sample["usage"]["input_tokens"] for sample in data if sample.get("usage")]),
            "output_tokens": sum([sample["usage"]["output_tokens"] for sample in data if sample.get("usage")]),
            "total_tokens": sum([sample["usage"]["total_tokens"] for sample in data if sample.get("usage")])
        },
        "cost": {
            "input": sum([sample["cost"]["input"] for sample in data if sample.get("cost")]),
            "output": sum([sample["cost"]["output"] for sample in data if sample.get("cost")]),
            "total": sum([sample["cost"]["total"] for sample in data if sample.get("cost")])
        }
    }

async def compute_gemini_aut_score(input_data, **kwargs):
    gemini_service = get_inference_service("gemini-3-flash-preview")
    num_batches = len(input_data["data"]) // 16 + 1
    for batch in tqdm(batched(input_data["data"], size=16), total=num_batches, desc="Computing Gemini AUT scores"):
        eval_samples = []
        for sample in batch:
            prompt = LLM_AUT_SCORING_TEMPLATE.format(stimuli=sample["stimuli"], output=sample["output"])
            eval_samples.append({"id": sample["id"], "user_prompt": prompt})
        
        responses = await gemini_service.generate(eval_samples)
        for sample, response in zip(batch, responses):
            try:
                score = int(response.text[0].strip())
                sample["gemini_aut_score"] = score
            except Exception as e:
                print(f"Error occurred while processing sample {sample['id']}: {e}")
                sample["gemini_aut_score"] = 0
    
    return {"gemini_aut_score": np.mean([sample["gemini_aut_score"] for sample in input_data["data"] if sample.get("gemini_aut_score")])}

METRIC_MAP = {
    "usage_and_cost": compute_usage_and_cost,
    "gemini_aut_score": compute_gemini_aut_score
}

async def report_metrics(data_files, metric=None, source_datapath=None):
    for data_file in data_files:
        input_data = read_json(data_file)
        if "data" in input_data:
            source_data_dict = {}
            if source_datapath:
                source_data = read_json(source_datapath)
                source_data_dict = {item["id"]: item for item in source_data["data"]}
            metrics = await METRIC_MAP[metric](input_data, source_data_dict=source_data_dict)
            if "metrics" not in input_data:
                input_data["metrics"] = {}
            input_data["metrics"].update(metrics)
            write_json(input_data, data_file)

async def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-paths", nargs="+", type=str, help="Path(s) to data", required=True)
    parser.add_argument("-m", "--metric", type=str, default="usage_and_cost", help="Metric name")
    parser.add_argument("--source-datapath", type=str, help="Path to source task data in json")

    args = parser.parse_args()

    files_to_process = []
    for data_path in args.data_paths:
        results_path = pathlib.Path(data_path)
        if results_path.is_file():
            files_to_process.append(data_path)
        else:
            files_to_process.extend(find_files(data_path))

    await report_metrics(files_to_process, metric=args.metric, source_datapath=args.source_datapath)


if __name__ == "__main__":
    asyncio.run(main())
