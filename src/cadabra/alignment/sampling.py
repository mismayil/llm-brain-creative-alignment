
import numpy as np
from abc import abstractmethod

from cadabra.model.modeling_utils import load_network_mask
from cadabra.utils import parse_sampling_strategy, apply_sampling_strategy
    
def parse_sampling_strategies(sampling) -> list[tuple[str, str]]:
    """
    Parse the sampling strategy string into time and layer strategies.
    # Example strategies:
    # "time:1::layer:-1" -> last layer at time point 1
    # "time:mean::layer:-1" -> last layer, mean over time
    # "time:mean::layer:3" -> layer 3, mean over time
    # "time:1:10::layer:mean" -> mean over layers from time point 1 to 10
    Returns:
        list[tuple[str, str]]: List of tuples containing strategy name and value.
    """
    sampling_parts = sampling.split("::")
    strategies = []
    for part in sampling_parts:
        name, value = parse_sampling_strategy(part)
        strategies.append((name, value))
    return strategies

class DataSampler:
    def __init__(self, sampling):
        self.sampling = sampling
        self.strategies = parse_sampling_strategies(sampling)

    @abstractmethod
    def sample(self):
        raise NotImplementedError

class BrainDataSampler(DataSampler):
    def __init__(self, sampling, noise_ceiling_data=None, noise_ceiling_threshold=0.0):
        super().__init__(sampling)
        self.noise_ceiling_data = noise_ceiling_data
        self.noise_ceiling_threshold = noise_ceiling_threshold

    def sample(self, data):
        """
        Sample brain data based on the specified strategy.
        Args:
            data (np.ndarray): Brain data with shape (T, D)
                               where T is number of time points (e.g. fmri TRs),
                               D is data dimension.
        Returns:
            np.ndarray: Sampled data with shape (T, D)
        """
        for dim, strategy in enumerate(self.strategies):
            data = apply_sampling_strategy(data, strategy, dim=dim)
        
        if self.noise_ceiling_data is not None:
            if self.noise_ceiling_data.shape[0] == data.shape[-1]:
                # Filter out data where noise ceiling is below the threshold if per voxel noise ceiling is provided
                mask = self.noise_ceiling_data > self.noise_ceiling_threshold
                data = data[:, mask]  # (T, num_voxels_above_threshold)
        return data

class ModelDataSampler(DataSampler):
    def __init__(self, sampling, network_path=None, network_type="network", ignore_first_layer=True):
        super().__init__(sampling)
        self.network_path = network_path
        self.network_type = network_type
        self.ignore_first_layer = ignore_first_layer
        self.network_mask = load_network_mask(network_path, network_type, ignore_first_layer) if network_path else None

    def sample(self, data) -> np.ndarray:
        """
        Sample model data based on the specified strategy.
        Args:
            data (np.ndarray): Model activations with shape (T, L, D)
                               where T is number of time points,
                               L is number of layers,
                               D is model dimension.
        Returns:
            np.ndarray: Sampled data with shape (T, L, D)
        """
        data = apply_sampling_strategy(data, self.strategies[0], dim=0)  # Apply time strategy first
        
        if self.network_mask is not None:
            # if network mask is provided, layer sampling is ignored
            assert data.shape[1] == self.network_mask.shape[0], "Data layers and network mask layers do not match"
            # Gather only the units specified by the network mask
            data = data[:, self.network_mask.astype(bool)]  # (T, num_active_units)
            data = np.expand_dims(data, axis=1)  # Add layer dimension back (T, 1, num_active_units)
        else:
            data = apply_sampling_strategy(data, self.strategies[1], dim=1)  # Apply layer strategy
        
        return data