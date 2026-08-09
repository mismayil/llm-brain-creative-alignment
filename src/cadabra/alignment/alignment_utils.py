from types import SimpleNamespace

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from scipy.stats import median_abs_deviation
import pathlib
import hashlib
import json

from cadabra.utils import json_serialize, read_json, get_wandb_runs

@dataclass
class BrainSample:
    subject: str
    stimuli: str
    data: np.ndarray # shape (num_tr, brain_x_dim, brain_y_dim, brain_z_dim)

@dataclass
class ModelSample:
    model_name: str
    stimuli: str
    data: np.ndarray # shape (num_tokens, num_layers, model_dim)
    subject: Optional[str] = None

@dataclass
class AlignmentResult:
    alignment_scores: np.ndarray  # shape (num_layers, num_voxels) or (num_layers, num_subjects)
    subject_alignment_scores: Optional[np.ndarray] = None  # shape (num_subjects, num_layers, num_voxels) if applicable
    subjects: Optional[List[str]] = None

def load_noise_ceiling_results(noise_ceiling_path: str, noise_ceiling_threshold: Optional[float] = None) -> Optional[dict]:
    if not noise_ceiling_path:
        return None
    noise_ceiling_results = read_json(noise_ceiling_path)
    noise_ceiling_data = np.load(pathlib.Path(noise_ceiling_path).parent / noise_ceiling_results["data"]["noise_ceiling_path"])
    if noise_ceiling_threshold is not None:
        noise_ceiling_mask = noise_ceiling_data > noise_ceiling_threshold
        noise_ceiling_data = noise_ceiling_data[noise_ceiling_mask]
        if noise_ceiling_results["metadata"]["subjects"]:
            noise_ceiling_results["metadata"]["subjects"] = [s for i, s in enumerate(noise_ceiling_results["metadata"]["subjects"]) if noise_ceiling_mask[i]]
    noise_ceiling_results["data"] = noise_ceiling_data
    print(f"Loaded noise ceiling data from {noise_ceiling_path} with shape {noise_ceiling_data.shape}")
    return noise_ceiling_results

def apply_noise_ceiling(alignment_results: AlignmentResult, noise_ceiling_results: dict):
    """
    Adjust alignment results based on noise ceiling data.
    If a noise ceiling for a voxel is zero or negative, 
    the corresponding alignment results is set to NaN.

    Parameters:
    - alignment_results: AlignmentResult, containing alignment scores of shape (num_tokens, num_layers, num_voxels) or (num_tokens, num_layers, num_subjects)
    - noise_ceiling_results: dict, containing metadata and data of np.ndarray of shape (num_voxels,) or (num_subjects,) depending on the noise ceiling type.

    Returns:
    - adjusted_alignment_scores: np.ndarray of same shape as alignment_results
    """
    noise_ceiling_data = noise_ceiling_results["data"]
    ns_type = noise_ceiling_results["metadata"]["noise_ceiling_type"]
    alignment_scores = alignment_results.alignment_scores
    print(f"Applying noise ceiling adjustment with data shape {noise_ceiling_data.shape}")
    if ns_type == "per_voxel":
        assert alignment_scores.shape[-1] == noise_ceiling_data.shape[0], "Alignment results and noise ceiling data must have the same number of voxels/subjects."
        adjusted_alignment_scores = np.zeros_like(alignment_scores)
        for voxel in range(alignment_scores.shape[-1]):
            ceiling = noise_ceiling_data[voxel]
            adjusted_alignment_scores[..., voxel] = np.clip(alignment_scores[..., voxel] / ceiling, a_min=-1, a_max=1) if ceiling > 0 else np.nan
    elif ns_type == "median_voxel":
        global_noise_ceiling = np.nanmedian(noise_ceiling_data).item()
        print(f"Using global noise ceiling value: {global_noise_ceiling}")
        adjusted_alignment_scores = np.clip(alignment_scores / global_noise_ceiling, a_min=-1, a_max=1)
    elif ns_type == "mean_voxel":
        global_noise_ceiling = np.nanmean(noise_ceiling_data).item()
        print(f"Using global noise ceiling value: {global_noise_ceiling}")
        adjusted_alignment_scores = np.clip(alignment_scores / global_noise_ceiling, a_min=-1, a_max=1)
    elif "per_subject" in ns_type:
        alignment_subjects = alignment_results.subjects
        noise_ceiling_subjects = noise_ceiling_results["metadata"]["subjects"]
        adjusted_alignment_scores = np.zeros_like(alignment_scores)
        for subj_idx, subject in enumerate(alignment_subjects):
            if subject in noise_ceiling_subjects:
                ceiling = noise_ceiling_data[noise_ceiling_subjects.index(subject)]
                adjusted_alignment_scores[..., subj_idx] = np.clip(alignment_scores[..., subj_idx] / ceiling, a_min=-1, a_max=1) if ceiling > 0 else np.nan
            else:
                adjusted_alignment_scores[..., subj_idx] = np.nan
    elif ns_type == "rsa":
        global_noise_ceiling = np.nanmedian(noise_ceiling_data).item()
        print(f"Using global noise ceiling value: {global_noise_ceiling}")
        adjusted_alignment_scores = np.clip(alignment_scores / global_noise_ceiling, a_min=-1, a_max=1)
    else:
        raise ValueError(f"Unknown noise ceiling type: {ns_type}")
    return adjusted_alignment_scores

def compute_layer_alignment_metrics(alignment_scores) -> dict:
    """
    Compute alignment metrics from the alignment scores.

    Parameters:
    - alignment_scores: np.ndarray of shape (num_layers, num_voxels) or (num_layers, num_subjects)

    Returns:
    - metrics: dict containing various alignment metrics
    """
    num_total_voxels = alignment_scores.shape[-1]
    num_nan_voxels = np.sum(np.isnan(alignment_scores), axis=-1)
    num_non_nan_voxels = num_total_voxels - num_nan_voxels
    num_invalid_voxels = num_nan_voxels.tolist() if isinstance(num_nan_voxels, np.ndarray) else int(num_nan_voxels)
    num_valid_voxels = num_non_nan_voxels.tolist() if isinstance(num_non_nan_voxels, np.ndarray) else int(num_non_nan_voxels)
    mean_pred = np.nan_to_num(np.nanmean(alignment_scores, axis=-1), nan=0.0)
    median_pred = np.nan_to_num(np.nanmedian(alignment_scores, axis=-1), nan=0.0)
    metrics = {
        "num_total_voxels": num_total_voxels,
        "num_valid_voxels": num_valid_voxels,
        "num_invalid_voxels": num_invalid_voxels,
        "mean_pred": mean_pred,
        "median_pred": median_pred,
        "median_abs_dev_pred": np.nan_to_num(median_abs_deviation(alignment_scores, axis=-1, nan_policy='omit'), nan=0.0),
        "std_pred": np.nan_to_num(np.nanstd(alignment_scores, axis=-1), nan=0.0),
        "max_pred": np.nan_to_num(np.nanmax(alignment_scores, axis=-1), nan=0.0),
        "min_pred": np.nan_to_num(np.nanmin(alignment_scores, axis=-1), nan=0.0)
    }

    # take the best layer per brain signal unit based on median predictivity
    median_pred = metrics["median_pred"]
    best_layer = np.argmax(median_pred, axis=0)
    metrics["best_layer"] = {
        "layer_num": int(best_layer),
        "mean_pred": metrics["mean_pred"][best_layer],
        "median_pred": metrics["median_pred"][best_layer],
        "median_abs_dev_pred": metrics["median_abs_dev_pred"][best_layer],
        "std_pred": metrics["std_pred"][best_layer],
        "max_pred": metrics["max_pred"][best_layer],
        "min_pred": metrics["min_pred"][best_layer]
    }
    metrics["last_layer"] = {
        "layer_num": len(median_pred),
        "mean_pred": metrics["mean_pred"][-1],
        "median_pred": metrics["median_pred"][-1],
        "median_abs_dev_pred": metrics["median_abs_dev_pred"][-1],
        "std_pred": metrics["std_pred"][-1],
        "max_pred": metrics["max_pred"][-1],
        "min_pred": metrics["min_pred"][-1]
    }

    return json_serialize(metrics)

def compute_alignment_metrics(alignment_scores: np.ndarray) -> list:
    """
    Compute alignment metrics from the alignment scores.

    Parameters:
    - alignment_scores: np.ndarray of shape (num_tokens, num_layers, num_voxels) or (num_tokens, num_layers, num_subjects)

    Returns:
    - metrics: list of dicts containing various alignment metrics for each token
    """
    token_alignment_metrics = []
    for token_idx in range(alignment_scores.shape[0]):
        token_scores = alignment_scores[token_idx]  # shape (num_layers, num_voxels) or (num_layers, num_subjects)
        token_metrics = compute_layer_alignment_metrics(token_scores)
        token_alignment_metrics.append(token_metrics)
    return token_alignment_metrics

CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".cache" / "wandb_runs"


def _build_runs_cache_key(cache_args):
    # Use a canonical JSON payload to ensure stable keys across equivalent calls.
    key_payload = json.dumps(cache_args, sort_keys=True, default=str)
    return hashlib.sha256(key_payload.encode("utf-8")).hexdigest()


def _serialize_runs(runs):
    return [
        {
            "config": run.config,
            "summary_metrics": run.summary_metrics,
        }
        for run in runs
    ]


def _deserialize_runs(serialized_runs):
    return [
        SimpleNamespace(
            config=run_data["config"],
            summary_metrics=run_data.get("summary_metrics", {}),
        )
        for run_data in serialized_runs
    ]

def get_alignment_runs(alignment_method="rsa_per_subject", nc_threshold=0.0, brain_network=None, dataset=None, model_data_sampling=None,
            model_network=None, network_pooling=None, activation_mode=None, regression_target=None, wandb_project="cadabra-alignments", force_refresh=False):
    cache_args = {
        "alignment_method": alignment_method,
        "nc_threshold": nc_threshold,
        "brain_network": brain_network,
        "dataset": dataset,
        "model_data_sampling": model_data_sampling,
        "model_network": model_network,
        "network_pooling": network_pooling,
        "activation_mode": activation_mode,
        "regression_target": regression_target,
        "wandb_project": wandb_project,
    }
    cache_key = _build_runs_cache_key(cache_args)
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists() and not force_refresh:
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache_payload = json.load(f)
            print(f"Cache hit for get_runs: {cache_path}")
            return _deserialize_runs(cache_payload.get("runs", []))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as err:
            print(f"Failed to read cache at {cache_path}. Re-querying wandb. Error: {err}")

    filters = {
        "config.config.alignment.alignment_method": alignment_method,
        "config.config.brain_args.noise_ceiling_threshold": nc_threshold,
        "config.config.brain_args.brain_datapath": {"$regex": brain_network},
        "config.config.model_args.model_datapath": {"$regex": dataset},
        "config.config.model_args.model_data_sampling": model_data_sampling,
    }

    if regression_target is not None:
        filters["config.config.alignment.alignment_args.regression_target"] = regression_target

    if model_network == "layers":
        filters["config.config.model_args.model_network_path"] = None
    else:
        filters["config.config.model_args.model_network_path"] = {"$regex": network_pooling}
        filters["config.config.model_args.model_network_type"] = model_network

    print(f"Querying wandb for runs with filters: {filters}")
    runs = get_wandb_runs(project=wandb_project, filters=filters)
    final_runs = []

    for run in runs:
        model_data = read_json(run.config["config"]["model_args"]["model_datapath"])
        prompt_only = model_data["metadata"]["config"]["prompt_only"]
        if activation_mode == "prompt" and prompt_only or activation_mode != "prompt" and not prompt_only:
            final_runs.append(run)
    
    print(f"Found {len(final_runs)} runs matching the criteria.")

    # group runs by config hash and check if there are any runs that have the same config, if so keep the latest one per group
    latest_runs = []
    runs_by_config = {}

    for run in final_runs:
        config_hash = hashlib.sha256(str(run.config["config"]).encode()).hexdigest()
        if config_hash not in runs_by_config:
            runs_by_config[config_hash] = []
        runs_by_config[config_hash].append(run)
    for config_hash, runs in runs_by_config.items():
        latest_run = max(runs, key=lambda r: r.created_at)
        latest_runs.append(latest_run)

    print(f"After keeping only the latest run per config, {len(latest_runs)} runs remain.")

    if latest_runs:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump({"cache_args": cache_args, "runs": _serialize_runs(latest_runs)}, f, default=str)
            print(f"Cached get_runs result at: {cache_path}")
        except OSError as err:
            print(f"Failed to write cache at {cache_path}. Continuing without cache. Error: {err}")

    return latest_runs