from typing import Optional
from nilearn import image as nimg
import bids
import os
import random
import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_yeo_2011
import pathlib
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from cadabra.utils import keep_most_frequent_size

DEFAULT_CONFOUND_VARS = ["trans_x", "trans_y", "trans_z",
                        "rot_x", "rot_y", "rot_z",
                        "global_signal",
                        "csf", "white_matter"]

def save_tr_value(data_dir, t_r):
    """
    Save the Repetition Time (TR) value to a text file in the data directory.
    
    Parameters:
    data_dir (str): Path to the BIDS dataset directory.
    t_r (float): The TR value in seconds.
    """
    tr_file_path = os.path.join(data_dir, "tr_value.txt")
    with open(tr_file_path, 'w') as f:
        f.write(f"{t_r}\n")
    print(f"TR value saved to {tr_file_path}")

def read_tr_value(data_dir):
    """
    Read the Repetition Time (TR) value from a text file in the data directory.
    
    Parameters:
    data_dir (str): Path to the BIDS dataset directory.
    
    Returns:
    float: The TR value in seconds, or None if the file does not exist.
    """
    tr_file_path = os.path.join(data_dir, "tr_value.txt")
    if os.path.exists(tr_file_path):
        with open(tr_file_path, 'r') as f:
            t_r = float(f.read().strip())
        print(f"TR value loaded from {tr_file_path}: {t_r} seconds")
        return t_r
    else:
        print(f"No TR value file found at {tr_file_path}.")
        return None

def get_tr_value(data_dir, space="MNI152NLin2009cAsym", task="dt", func_desc="preproc"):
    """
    Get the Repetition Time (TR) value from one of the functional files' metadata.
    
    Parameters:
    data_dir (str): Path to the BIDS dataset directory.
    space (str): The spatial normalization space used.
    task (str): The task name.
    func_desc (str): The functional description.
    
    Returns:
    float: The TR value in seconds.
    """
    tr_value = read_tr_value(data_dir)
    if tr_value is not None:
        return tr_value

    print("Identifying the TR value from metadata")
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    func_files = layout.get(datatype="func", task=task, desc=func_desc,
                            space=space, extension="nii.gz", return_type="file")
    if not func_files:
        print("Warning: No functional files provided to extract TR value.")
        return None

    sample_func_files = random.sample(func_files, min(5, len(func_files)))
    
    tr_values = []

    for func_file in sample_func_files:
        metadata = layout.get_metadata(func_file)
        if metadata and "RepetitionTime" in metadata:
            tr_values.append(metadata["RepetitionTime"])

    if tr_values:
        if set(tr_values) == {tr_values[0]}:
            tr_value = tr_values[0]
            save_tr_value(data_dir, tr_value)
            return tr_value
        else:
            raise ValueError("Inconsistent TR values found across functional files.")
    
    print("No TR value found in metadata, checking NIfTI headers.")
    tr_values = []

    for func_file in sample_func_files:
        func_img = nimg.load_img(func_file)
        tr_values.append(func_img.header.get_zooms()[-1])
    
    if tr_values:
        if set(tr_values) == {tr_values[0]}:
            tr_value = tr_values[0]
            save_tr_value(data_dir, tr_value)
            return tr_value
        else:
            raise ValueError("Inconsistent TR values found across functional files.")

def get_yeo_parcellation(networks=7):
    """
    Get Yeo et al. 2011 network parcellation.
    
    7-network parcellation:
    1-Visual, 2-Somatomotor, 3-Dorsal Attention, 4-Ventral Attention, 
    5-Limbic, 6-Frontoparietal, 7-Default
    
    17-network parcellation:
    1-Visual A, 2-Visual B, 3-Somatomotor A, 4-Somatomotor B,
    5-Dorsal Attention A, 6-Dorsal Attention B, 7-Salience/Ventral Attention A,
    8-Salience/Ventral Attention B, 9-Limbic A, 10-Limbic B,
    11-Frontoparietal A, 12-Frontoparietal B, 13-Frontoparietal C,
    14-Default A, 15-Default B, 16-Default C, 17-Temporal Parietal
    
    Parameters:
    -----------
    networks : int
        Either 7 for 7-network or 17 for 17-network parcellation
        
    Returns:
    --------
    parcellation : nibabel.Nifti1Image
        The parcellation image
    """
    atlas = fetch_atlas_yeo_2011(n_networks=networks)
    return nib.load(atlas.maps)

def extract_network_maps(networks=7, parcellation="yeo"):
    """
    Extract individual network maps from the specified parcellation.
    
    Parameters:
    -----------
    networks : int
        Either 7 for 7-network or 17 for 17-network parcellation

    parcellation : str
        The name of the parcellation to use (e.g. "yeo")
        Currently, only "yeo" is implemented.
    Returns:
    --------
    network_maps : dict
        Dictionary with network indices as keys and nibabel images as values
    """
    if parcellation == "yeo":
        parcellation_img = get_yeo_parcellation(networks)
    else:
        raise ValueError("Unsupported parcellation type.")

    parcellation_data = parcellation_img.get_fdata()
    
    network_maps = {}
    
    for i in range(1, networks + 1):
        network_mask = (parcellation_data == i).astype(np.float32)
        network_img = nib.Nifti1Image(
            network_mask, 
            parcellation_img.affine, 
            parcellation_img.header
        )
        network_maps[i] = network_img
    
    return network_maps

YEO7_NETWORK_IDX_MAP = {
    "vis_a": 1,
    "som": 2,
    "dors_attn": 3,
    "vent_attn": 4,
    "limbic": 5,
    "fp": 6,
    "dmn": 7
}

YEO17_NETWORK_IDX_MAP = {
    "vis_a": 1,
    "vis_b": 2,
    "som_a": 3,
    "som_b": 4,
    "dors_attn_a": 5,
    "dors_attn_b": 6,
    "sal_vent_attn_a": 7,
    "sal_vent_attn_b": 8,
    "limbic_a": 9,
    "limbic_b": 10,
    "fp_a": 11,
    "fp_b": 12,
    "fp_c": 13,
    "dmn_a": 14,
    "dmn_b": 15,
    "dmn_c": 16,
    "temp_par": 17
}

def load_roi_mask(roi: str = "yeo:dmn", roi_path: Optional[str] = None, roi_threshold: float = 0.0) -> nib.Nifti1Image:
    """
    Load ROI network mask.
    Parameters:
    -----------
    roi : str
        ROI specification string.
        Examples:
        "whole_brain" -> no masking
        "yeo:dmn" -> load Yeo 7-network DMN mask
        "yeo17:fp_a" -> load Yeo 17-network Frontoparietal A mask
    roi_path : str or None
        Path to a custom ROI mask file (NIfTI format). If provided, this will override the `roi` parameter.
    roi_threshold : float
        Threshold for binarizing continuous ROI masks.
    Returns:
    --------
    network_img : nib.Nifti1Image or None
        The network mask image, or None if whole_brain is specified
    """
    if roi_path:
        if not os.path.exists(roi_path):
            raise ValueError(f"Provided ROI path does not exist: {roi_path}")
        print(f"Loading custom ROI mask from: {roi_path}")
        network = nib.load(roi_path)
        network_data = network.get_fdata()
        if np.issubdtype(network_data.dtype, np.floating):
            print(f"Binarizing continuous ROI mask with threshold: {roi_threshold}")
            network_data = (network_data >= roi_threshold).astype(np.float32)
            network = nib.Nifti1Image(network_data, network.affine, network.header)
        return network

    if roi == "whole_brain":
        return None
    
    if roi.startswith("yeo"):
        parts = roi.split(":")
        if len(parts) != 2:
            raise ValueError("Invalid Yeo ROI specification.")
        
        parcellation = parts[0]
        network_name = parts[1]
        network_idx_map = None
        networks = None

        if parcellation == "yeo":
            network_idx_map = YEO7_NETWORK_IDX_MAP
            networks = 7
        elif parcellation == "yeo17":
            network_idx_map = YEO17_NETWORK_IDX_MAP
            networks = 17
        else:
            raise ValueError("Unsupported Yeo parcellation type.")
        
        network_maps = extract_network_maps(networks=networks, parcellation="yeo")

        if network_name not in network_idx_map:
            raise ValueError(f"Unknown network name: {network_name}")
        network_idx = network_idx_map[network_name]
        if network_idx not in network_maps:
            raise ValueError(f"Network index {network_idx} not found in extracted maps.")
        network_img = network_maps[network_idx]
    else:
        raise ValueError("Unsupported ROI specification.")
    return network_img

def affine_mismatch(img1: nib.Nifti1Image, img2: nib.Nifti1Image, tol: float = 1e-5) -> bool:
    """
    Check if two NIfTI images have mismatched affines.
    
    Parameters:
    -----------
    img1 : nib.Nifti1Image
        First image
    img2 : nib.Nifti1Image
        Second image
    tol : float
        Tolerance for affine comparison
    
    Returns:
    --------
    bool
        True if affines mismatch, False otherwise
    """
    return not np.allclose(img1.affine, img2.affine, atol=tol)

def apply_roi_mask(brain_img: nib.Nifti1Image, roi_mask: nib.Nifti1Image, 
                   resample: bool = False, interpolation: str = "nearest") -> np.ndarray:
    """
    Apply ROI mask to brain image data.
    Parameters:
    -----------
    brain_img : nib.Nifti1Image
        The brain image data
    roi_mask : nib.Nifti1Image or None
        The ROI mask image, or None for whole brain
    resample : bool
        Whether to resample the ROI mask to match the brain image
    interpolation : str
        Interpolation method for resampling (default: "nearest")
    Returns:
    --------
    masked_data : np.ndarray
        The masked brain data as a 2D array (n_timepoints, n_voxels) or (T, V)
    """
    if not roi_mask:
        return brain_img.get_fdata().reshape(-1, brain_img.shape[-1]).T  # (T, V)

    if resample or affine_mismatch(brain_img, roi_mask):
        roi_mask = resample_to_img(
            roi_mask,               # the network mask image
            brain_img,              # reference image (same space/resolution as data)
            interpolation=interpolation, # important: preserve labels
            copy_header=True,         # and copy header info
            force_resample=True
        )
    roi_img = apply_mask(brain_img, roi_mask)
    return roi_img

def load_brain_neural_data(brain_datapath: str, brain_neural_datapath: str) -> np.ndarray:
    """ Load brain neural data from a given path. """
    brain_data_dir = pathlib.Path(brain_datapath)
    if brain_data_dir.is_file():
        brain_data_dir = brain_data_dir.parent
    brain_neural_data = np.load(brain_data_dir / brain_neural_datapath) # shape (num_tr, brain_x_dim, brain_y_dim, brain_z_dim) or (num_tr, num_voxels)
    return brain_neural_data

def filter_dim_mismatch_samples(brain_data, data_field: str = "fmri_data", return_indices: bool = False) -> list | tuple:
    # Remove samples that have mismatching fMRI data lengths
    fmri_samples = [sample[data_field] for sample in brain_data]
    _, frequent_indices = keep_most_frequent_size(fmri_samples, return_indices=True)
    brain_data = [sample for i, sample in enumerate(brain_data) if i in frequent_indices]
    if return_indices:
        return brain_data, frequent_indices
    return brain_data