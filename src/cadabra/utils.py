from collections import Counter
import json
import uuid
import pandas as pd
import glob
from string import Formatter
import hashlib
import pathlib
from typing import List, Optional, Tuple
import numpy as np
import os
import wandb
import time
import yaml
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def write_json(data, path, ensure_ascii=True, indent=4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def generate_unique_id():
    return str(uuid.uuid4()).split("-")[-1]

def convert_nan_to_none(data):
    if isinstance(data, list):
        return [convert_nan_to_none(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_nan_to_none(value) for key, value in data.items()}
    elif isinstance(data, float) and pd.isna(data):  # Check for NaN
        return None
    else:
        return data

def get_template_keys(template):
    return [i[1] for i in Formatter().parse(template) if i[1] is not None]

def find_files(directory, extension="json"):
    return glob.glob(f"{directory}/**/*.{extension}", recursive=True)

def remainder_args_to_dict(remainder_args):
    args_dict = {}
    key = None
    for arg in remainder_args:
        if arg == "--":
            continue
        if arg.startswith("--"):
            if key is not None:
                args_dict[key] = True  # Flag without value
            key = arg.lstrip("-")
        else:
            if key is not None:
                args_dict[key] = arg
                key = None
    if key is not None:
        args_dict[key] = True  # Last flag without value
    return args_dict

def batched(lst, size=4):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def none_or_float(value):
    if value.lower() == "none":
        return None
    return float(value)

def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value

def generate_datetime_id():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_hash_id(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

def read_data(datapath, extension="json", separator=None):
    if pathlib.Path(datapath).is_file():
        if datapath.endswith(f'.{extension}'):
            if extension == 'json':
                return read_json(datapath)
            elif extension == 'jsonl':
                return read_jsonl(datapath)
            elif extension == 'csv':
                return pd.read_csv(datapath, sep=separator)
            elif extension == 'xlsx':
                return pd.read_excel(datapath, engine='openpyxl')
            else:
                raise ValueError("Unsupported file format. Only .json, .jsonl, .csv and .xlsx are supported.")
        else:
            raise ValueError("Unsupported file format. Only .json, .jsonl, .csv and .xlsx are supported.")
    elif pathlib.Path(datapath).is_dir():
        all_files = find_files(datapath, extension=extension)
        all_data = []
        for file in all_files:
            all_data.append(read_data(file, extension=extension))
        return all_data
    else:
        raise ValueError("The provided datapath is neither a file nor a directory.")

def keep_most_frequent_size(data: List[np.ndarray], dim: int = -1, return_indices: bool = False) -> List[np.ndarray] | Tuple[np.ndarray, List[int]]:
    """ Keep only the data samples with the most frequent dimension size. """
    dims = [d.shape[dim] for d in data]
    dim_counts = Counter(dims)
    indices = list(range(len(data)))
    if len(dim_counts) > 1:
        print(f"Warning: found different dimensions: {dim_counts}. Keeping the most frequent one.")
        most_frequent_dim = dim_counts.most_common(1)[0][0]
        indices = [i for i, d in enumerate(dims) if d == most_frequent_dim]
        data = [d for d in data if d.shape[dim] == most_frequent_dim]
        print(f"After filtering, {len(data)} samples remain with dimension {most_frequent_dim}.")
    if return_indices:
        return data, indices
    return data

def is_immutable(obj):
    return isinstance(obj, (str, int, float, bool, tuple, type(None)))

def cache(cache_dict):
    def decorator_cache(func):
        def wrapper(*args, **kwargs):
            if all(is_immutable(arg) for arg in args) and all(
                is_immutable(val) for val in kwargs.values()
            ):
                key = (args, frozenset(kwargs.items()))
                if key in cache_dict:
                    return cache_dict[key]
                result = func(*args, **kwargs)
                cache_dict[key] = result
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator_cache

MODEL_COSTS = {
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
    "gpt-4": {"input": 30e-6, "output": 60e-6},
    "gpt-4o": {"input": 2.5e-6, "output": 10e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
    "gpt-4-0125-preview": {"input": 10e-6, "output": 30e-6},
    "gpt-4o-2024-08-06": {"input": 2.5e-6, "output": 10e-6},
    "text-davinci-003": {"input": 0.00002, "output": 0.00002},
    "gemini-1.5-flash": {"input": 3.5e-7, "output": 1.05e-6},
    "gemini-1.5-pro": {"input": 3.5e-6, "output": 10.5e-6},
    "claude-3-5-sonnet-20240620": {"input": 3e-6, "output": 15e-6},
    "claude-3-5-haiku-20241022": {"input": 1e-6, "output": 5e-6},
    "claude-3-opus-20240229": {"input": 15e-6, "output": 75e-6},
    "claude-3-sonnet-20240229": {"input": 3e-6, "output": 15e-6},
    "claude-3-haiku-20240307": {"input": 0.25e-6, "output": 1.25e-6},
}

MODEL_ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "gpt-4o": "o200k_base",
}

def num_tokens_from_string(text, model):
    import tiktoken

    if model not in MODEL_ENCODINGS:
        return 0
    encoding_name = MODEL_ENCODINGS[model]
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def compute_usage(
    sample,
    model,
    input_attrs=["system_prompt", "user_prompt"],
    output_attrs=["output"],
    max_input_tokens=None,
    max_output_tokens=None,
):
    if model not in MODEL_COSTS:
        return None, None

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    usage = sample.get("usage")

    if not usage:
        input_tokens = 0
        output_tokens = 0

        if max_input_tokens:
            input_tokens = max_input_tokens
        else:
            for attr in input_attrs:
                if attr in sample:
                    input_tokens += num_tokens_from_string(sample[attr], model)

        if max_output_tokens:
            output_tokens = max_output_tokens
        else:
            for attr in output_attrs:
                if attr in sample:
                    output_tokens += num_tokens_from_string(sample[attr], model)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    input_cost = usage["input_tokens"] * MODEL_COSTS[model]["input"]
    output_cost = usage["output_tokens"] * MODEL_COSTS[model]["output"]

    return usage, {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


def none_or_str(value):
    if value.lower() == "none":
        return None
    return value


def wandb_log_run(name, config=None, metrics=None, project=None, run_id=None):
    if run_id is not None:
        run = get_wandb_run(run_id, project=project)
        if run:
            run.delete()
    run = wandb.init(name=name, project=project, config=config)
    if isinstance(metrics, list):
        for metric in metrics:
            run.log(metric)
    else:
        run.log(metrics)
    run.finish()
    return run


def get_wandb_run(run_id, entity=None, project=None):
    entity = os.getenv("WANDB_ENTITY") if entity is None else entity
    project = os.getenv("WANDB_PROJECT") if project is None else project
    try:
        return wandb.Api().run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"Error getting run {run_id}: {str(e)}")
        return None

def get_wandb_runs(entity=None, project=None, filters=None):
    entity = os.getenv("WANDB_ENTITY") if entity is None else entity
    project = os.getenv("WANDB_PROJECT") if project is None else project
    return wandb.Api().runs(f"{entity}/{project}", filters=filters)

def prepare_metrics_for_wandb(metrics, exclude_prefixes=None):
    if exclude_prefixes is None:
        exclude_prefixes = []

    wandb_metrics = {}

    if isinstance(metrics, list):
        prepared_metrics = []
        for metric in metrics:
            prepared_metric = prepare_metrics_for_wandb(metric, exclude_prefixes)
            prepared_metrics.append(prepared_metric)
        return prepared_metrics

    for key, value in metrics.items():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        if isinstance(value, dict):
            wandb_metrics[key] = prepare_metrics_for_wandb(value, exclude_prefixes)
        else:
            if is_immutable(value):
                wandb_metrics[key] = value
            else:
                value_array = np.asarray(value)
                if value_array.ndim == 1:
                    wandb_metrics[key] = value_array.mean().item()
                    wandb_metrics[f"{key}_std"] = value_array.std().item()

    return wandb_metrics

def collect_wandb_table_metrics(
    metrics,
    prefix="",
    exclude_prefixes=None,
):
    """
    Recursively traverse a nested metrics dict and return a dict
    of {flat_key: list}, where flat_key is constructed by
    joining nested keys with '.' (e.g. 'cv.mean_linear_pred').
    """
    if exclude_prefixes is None:
        exclude_prefixes = []

    tables = {}

    for key, value in metrics.items():
        if any(key.startswith(p) for p in exclude_prefixes):
            continue

        flat_key = f"{prefix}.{key}" if prefix else key

        # Recurse into sub-dicts
        if isinstance(value, dict):
            sub_tables = collect_wandb_table_metrics(
                value,
                prefix=flat_key,
                exclude_prefixes=exclude_prefixes,
            )
            tables.update(sub_tables)
            continue

        # Only convert non-scalar array-like values
        if is_immutable(value):
            continue

        value_array = np.asarray(value)

        if value_array.ndim != 1:
            continue

        tables[flat_key] = value_array.tolist()

    return tables

def create_wandb_tables(tables_dict):
    """
    Given a dict of {metric_name: list_of_values}, create
    wandb.Table where each metric name is a column.
    """
    if not tables_dict:
        return None

    df = pd.DataFrame(tables_dict)
    wandb_table = wandb.Table(dataframe=df)
    return wandb_table

def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers in a 1D array using the IQR method.

    Parameters:
        data (array-like): The input data.
        multiplier (float): The IQR multiplier, usually 1.5 (can use 3.0 for more extreme outliers).

    Returns:
        outlier_indices (list): Indices of outlier elements in the data.
    """
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_indices

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"'{func.__name__}' took {elapsed:.6f} seconds")
        return result
    return wrapper

def json_serialize(obj) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return [json_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {key: json_serialize(value) for key, value in obj.items()}
    return obj

def compute_rsa(data1: np.ndarray, data2: np.ndarray, **kwargs) -> Tuple[float, float]:
    data1_rdm = squareform(pdist(data1, metric='correlation'))
    data2_rdm = squareform(pdist(data2, metric='correlation'))
    n = data1_rdm.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    data1_rdm_flat = data1_rdm[mask].flatten()
    data2_rdm_flat = data2_rdm[mask].flatten()
    rsa, pvalue = spearmanr(data1_rdm_flat, data2_rdm_flat)
    return rsa, pvalue

def get_stimuli_id(sample):
    stimuli = sample.get("stimuli", sample.get("stimuli_id"))
    if stimuli is None:
        if "templeton_aut-create" in sample.get("id", ""):
            stimuli = sample["id"].replace("templeton_aut-create-", "").replace("_", " ")
        elif "templeton_aut-object" in sample.get("id", ""):
            stimuli = sample["id"].replace("templeton_aut-object-", "").replace("_", " ")
    return str(stimuli)

def get_subject_id(sample):
    return str(sample.get("subject_id", sample.get("subject", "N/A")))

def parse_sampling_strategy(sampling: str) -> tuple:
    """
    Parse a sampling strategy string into its name and value.
    Args:
        sampling (str): A sampling strategy string, e.g. "time:1", "time:mean", "layer:-1", 
                        "layer:mean", "layer:3", "time:1:10".
    Returns:
        tuple: (strategy_name, strategy_value), e.g. ("time", 1), ("layer", "mean").
    """
    parts = sampling.split(":")
    name = parts[0]
    value = parts[1:]

    if len(value) == 1:
        value = value[0]
        try:
            value = int(value)
        except ValueError:
            pass
        return (name, value)

    start_idx = int(value[0])
    stop_idx = int(value[1]) if len(value) > 1 and value[1] != "" else None
    step_idx = int(value[2]) if len(value) > 2 and value[2] != "" else None
    value = slice(start_idx, stop_idx, step_idx)

    return (name, value)

def apply_sampling_strategy(data: np.ndarray, strategy: tuple, dim: int = 0, keep_dims: bool = True) -> np.ndarray:
    """
    Apply a sampling strategy to the data.
    Args:
        data (np.ndarray): Input data to sample from.
        strategy (tuple): A tuple containing the strategy name and value, e.g. ("time", 1), ("layer", "mean").
        dim (int): The dimension along which to apply the sampling strategy.
        keep_dims (bool): Whether to keep the dimensions when computing the mean.
    Returns:
        np.ndarray: Sampled data based on the strategy.
    """
    strategy_name, strategy_value = strategy
    if strategy_value == "mean":
        return np.mean(data, axis=dim, keepdims=keep_dims)
    elif strategy_value == "all":
        return data
    elif isinstance(strategy_value, int):
        if keep_dims:
            return data.take(indices=[strategy_value], axis=dim)
        return data.take(indices=strategy_value, axis=dim)
    elif isinstance(strategy_value, list):
        return data.take(indices=strategy_value, axis=dim)
    elif isinstance(strategy_value, slice):
        return data.take(indices=range(*strategy_value.indices(data.shape[dim])), axis=dim)
    else:
        raise ValueError(f"Unsupported sampling strategy value: {strategy_value}")

def extract_model_size(model_name: str) -> Optional[int]:
    import re
    match = re.search(r'(\d+)([mMbB]{1})', model_name)
    if match:
        size = int(match.group(1))
        suffix = match.group(2)
        if suffix in ['m', 'M']:
            size *= 1e6
        elif suffix in ['b', 'B']:
            size *= 1e9
        return int(size)
    return None

def get_dict_value(obj, key_path, default=None):
    keys = key_path.split(".")
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def create_wandb_filters_from_config(config, key_prefix="config"):
    filters = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sub_filters = create_wandb_filters_from_config(value, key_prefix=f"{key_prefix}.{key}")
            filters.update(sub_filters)
        else:
            filters[f"{key_prefix}.{key}"] = value
    return filters