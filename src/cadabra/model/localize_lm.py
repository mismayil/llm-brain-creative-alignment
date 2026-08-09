from typing import List, Optional

import pathlib
import torch
import argparse
import numpy as np
from scipy.stats import ttest_ind, false_discovery_control
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from cadabra.utils import read_json, write_json, generate_datetime_id
from cadabra.model.modeling_utils import load_model_neural_data

def pool_activations(
    activations: torch.Tensor,
    pooling: str = "last-token",
):
    """
    Pool activations across the sequence dimension.
    Arguments:
        activations: (S, L, D) array of activations
        pooling: pooling method, one of ["last-token", "mean", "sum"]
    Returns:
        pooled_activations: (L, D) array of pooled activations
    """
    if pooling == "last-token":
        pooled_activations = activations[-1]  # (L, D)
    elif pooling == "mean":
        pooled_activations = torch.mean(torch.tensor(activations), dim=0)  # (L, D)
    elif pooling == "sum":
        pooled_activations = torch.sum(torch.tensor(activations), dim=0)  # (L, D)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
    
    return pooled_activations  # (L, D)

def localize(
        pos_activations_lst: List[np.ndarray],
        neg_activations_lst: List[np.ndarray],
        pooling: str = "last-token",
        num_units: Optional[int] = None, 
        percentage: float = 1,
        localize_range: str = "100-100",
        seed: int = 42,
) -> np.ndarray:
    """
    Localize task-selective units in the model.
    Arguments:
        pos_activations_lst: list of (S, L, D) arrays of positive activations
        neg_activations_lst: list of (S, L, D) arrays of negative activations
        pooling: pooling method, one of ["last-token", "mean", "sum"]
        num_units: number of units to localize
        percentage: percentage of units to localize
        localize_range: percentile range to localize, e.g., "90-100"
        seed: random seed
    Returns:
        network_mask: (L, D) binary array indicating localized units
    """
    range_start, range_end = map(int, localize_range.split("-"))
    pos_activations = np.stack([pool_activations(actv, pooling=pooling) for actv in pos_activations_lst])
    neg_activations = np.stack([pool_activations(actv, pooling=pooling) for actv in neg_activations_lst])
    num_layers, hidden_dim = pos_activations.shape[1], pos_activations.shape[2]

    print(f"Positive activations shape: {pos_activations.shape}")
    print(f"Negative activations shape: {neg_activations.shape}")

    p_values_matrix = np.zeros((num_layers, hidden_dim))
    t_values_matrix = np.zeros((num_layers, hidden_dim))

    for layer in tqdm(range(num_layers), desc="Localizing layers"):
        positive_actv = np.abs(pos_activations[:, layer, :])
        negative_actv = np.abs(neg_activations[:, layer, :])
        t_values_matrix[layer], p_values_matrix[layer] = ttest_ind(positive_actv, negative_actv, axis=0, equal_var=False)
    
    # replace nan t-values with 0.0
    t_values_matrix = np.nan_to_num(t_values_matrix, nan=0.0)

    # replace nan p-values with 1.0
    p_values_matrix = np.nan_to_num(p_values_matrix, nan=1.0)

    def is_topk(a, k=1):
        _, rix = np.unique(-a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    def is_bottomk(a, k=1):
        _, rix = np.unique(a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    np.random.seed(seed)

    if percentage is not None:
        num_units = int((percentage/100) * hidden_dim*num_layers)
        print(f"Percentage: {percentage}% --> Num Units: {num_units}")

    if localize_range is not None and range_start < range_end:
        range_start_val = np.percentile(t_values_matrix, range_start)
        range_end_val = np.percentile(t_values_matrix, range_end)
        # take random num_units from that percentile range
        mask_range = (t_values_matrix >= range_start_val) & (t_values_matrix <= range_end_val)
        total_num_units = np.prod(mask_range.shape)
        mask_range_indices = np.arange(total_num_units)[mask_range.flatten()]
        rand_indices = np.random.choice(mask_range_indices, size=num_units, replace=False)
        network_mask = np.full(total_num_units, 0)
        network_mask[rand_indices] = 1
        network_mask = network_mask.reshape(mask_range.shape)
        print(f"Num units in range {range_start}-{range_end}: {network_mask.sum()}")
    elif localize_range and range_start == range_end and int(range_start) == 0:
        network_mask = is_bottomk(t_values_matrix, k=num_units)
    else:
        network_mask = is_topk(t_values_matrix, k=num_units)

    print(f"Final num units: {network_mask.sum()}")
    adjusted_p_values = false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((num_layers, hidden_dim))

    return network_mask, adjusted_p_values

def localize_lm(config: DictConfig):
    assert config.percentage or config.num_units, "You must either provide percentage of units to localize or number of units"

    pos_activations_data = read_json(config.pos_activations_path)
    neg_activations_data = read_json(config.neg_activations_path)
    pos_activations = [load_model_neural_data(config.pos_activations_path, sample["activations_path"]) for sample in tqdm(pos_activations_data["data"], desc="Loading positive activations")]
    neg_activations = [load_model_neural_data(config.neg_activations_path, sample["activations_path"]) for sample in tqdm(neg_activations_data["data"], desc="Loading negative activations")]

    print(f"Loaded {len(pos_activations)} positive and {len(neg_activations)} negative activation samples")
    print(f"Localizing units with pooling: {config.pooling}, num_units: {config.num_units}, percentage: {config.percentage}, localize_range: {config.localize_range}, seed: {config.seed}")
    
    network_mask, adjusted_p_values = localize(
        pos_activations,
        neg_activations,
        pooling=config.pooling,
        num_units=config.num_units,
        percentage=config.percentage,
        localize_range=config.localize_range,
        seed=config.seed,
    )

    run_id = generate_datetime_id()
    num_localized_units = int(network_mask.sum())
    output_dir = pathlib.Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{run_id}_r{config.localize_range}_p{config.pooling}_u{num_localized_units}"
    if config.percentage:
        suffix += f"_pct{config.percentage}"
    output_path = output_dir / f"localization_{suffix}.json"
    mask_output_path = output_dir / f"network_mask_{suffix}.npy"
    p_values_path = output_dir / f"adjusted_p_values_{suffix}.npy"
    np.save(mask_output_path, network_mask)
    np.save(p_values_path, adjusted_p_values)

    print(f"Saved network mask to {mask_output_path}")
    print(f"Saved adjusted p-values to {p_values_path}")

    outputs = {
        "metadata": {
            "output_path": str(output_path),
            "run_id": run_id,
            "output_dir": str(output_dir),
            "config": OmegaConf.to_container(config, resolve=True),
        },
        "data": {
            "network_mask_path": str(mask_output_path.name),
            "num_localized_units": num_localized_units,
            "adjusted_p_values_path": str(p_values_path.name)
        }
    }

    write_json(outputs, output_path)
    print(f"Saved localization results to {output_path}")
    return output_path

@hydra.main(version_base=None, config_path="configs", config_name="localize_lm")
def main(config: DictConfig):
    localize_lm(config) 

if __name__ == "__main__":
    main()