import argparse
from nilearn import image as nimg
import os
import nibabel as nib
import numpy as np
import pandas as pd
import bids
from tqdm import tqdm
import concurrent.futures
from functools import partial

from cadabra.brain.fmri_utils import get_tr_value

def extract_task_volumes_with_nan_adjustment(func_image, events, nan_indices=None, tr=2.0, condition="create"):
    """
    Extract task trial volumes accounting for NaN volumes that were removed during cleaning.
    """
    if nan_indices is None:
        nan_indices = []
    
    # Filter for condition trials
    task_trials = events[events["trial_type"] == condition]
    
    # Convert onset times to original volume indices
    task_trials = task_trials.copy()
    task_trials["original_volume_start"] = np.floor(task_trials["onset"] / tr).astype(int)
    task_trials["original_volume_end"] = np.floor((task_trials["onset"] + task_trials["duration"]) / tr).astype(int)
    
    # Create mapping from original to task volume indices
    def map_original_to_task_volume(original_vol, nan_indices):
        """Map original volume index to task volume index after NaN removal."""
        # Count how many NaN volumes come before this volume
        nan_before = sum(1 for nan_idx in nan_indices if nan_idx < original_vol)
        return original_vol - nan_before
    
    # Get all original volumes to extract
    original_volumes_to_extract = []
    for _, trial in task_trials.iterrows():
        original_volumes_to_extract.extend(range(trial["original_volume_start"], trial["original_volume_end"]))
    
    original_volumes_to_extract = sorted(set(original_volumes_to_extract))
    
    # Filter out volumes that were removed due to NaN
    valid_original_volumes = [vol for vol in original_volumes_to_extract if vol not in nan_indices]
    
    # Map to task volume indices
    task_volumes_to_extract = [map_original_to_task_volume(vol, nan_indices) for vol in valid_original_volumes]
    
    print(f"Original volumes for task trials: {len(original_volumes_to_extract)}")
    print(f"Valid volumes after NaN removal: {len(valid_original_volumes)}")
    print(f"Task volume indices: {task_volumes_to_extract}")
    
    data = func_image.get_fdata()
    
    print(f"Task data shape: {data.shape}")
    
    # Validate task volume indices
    max_task_vol = max(task_volumes_to_extract) if task_volumes_to_extract else 0
    if max_task_vol >= data.shape[3]:
        print(f"Warning: Maximum task volume index ({max_task_vol}) exceeds data dimensions ({data.shape[3]})")
        task_volumes_to_extract = [vol for vol in task_volumes_to_extract if vol < data.shape[3]]
    
    # Extract task trial volumes
    if task_volumes_to_extract:
        task_data = data[:, :, :, task_volumes_to_extract]
        print(f"Extracted task data shape: {task_data.shape}")
        task_img = nib.Nifti1Image(task_data, func_image.affine, func_image.header)
        return task_volumes_to_extract, valid_original_volumes, task_img
    else:
        print("No valid task trial volumes found after NaN adjustment.")
        return [], [], None

def extract_fmri_task_data_for_subject(sub, data_dir, task="dt", space="MNI152NLin2009cAsym",
                                       t_r=2.0, func_desc="preproc", condition="create", func_suffix="cleaned"):
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    func_files = layout.get(subject=sub, datatype="func", task=task, desc=func_desc,
                            space=space, extension="nii.gz", suffix=func_suffix, return_type="file")
    events_files = layout.get(subject=sub, datatype="func", task=task, suffix="events",
                                extension="tsv", return_type="file")

    if not (len(func_files) == len(events_files)):
        print(f"Warning: Mismatch in number of functional, and events files for sub-{sub}. Skipping this subject.")
        return

    sub_dir_path = os.path.join(data_dir, f"sub-{sub}", "func")

    for func_file, events_file in zip(func_files, events_files):
        original_filename = os.path.basename(func_file)
        try:
            original_image = nimg.load_img(func_file)
        except Exception as e:
            print(f"Error loading functional image for sub-{sub}: {func_file}. Error: {e}")
            continue
        events = pd.read_csv(events_file, delimiter="\t")
        nan_row_indices = []

        nan_indices_file = os.path.join(sub_dir_path, f"sub-{sub}_nan_indices.txt")
        if os.path.exists(nan_indices_file):
            with open(nan_indices_file, "r") as f:
                nan_row_indices = [int(line.strip()) for line in f.readlines()]
            print(f"Loaded NaN indices for sub-{sub} from {nan_indices_file}")
        else:
            print(f"No NaN indices file found for sub-{sub}, assuming no NaNs.")

        # Extract task trial volumes
        task_volumes, original_volumes, task_img = extract_task_volumes_with_nan_adjustment(
            original_image, events, nan_indices=nan_row_indices, tr=t_r, condition=condition)

        if task_img:
            # Save extracted data
            task_filename = original_filename.replace(".nii.gz", f"_trial_{condition}.nii.gz")
            task_output_filepath = os.path.join(sub_dir_path, task_filename)
            nib.save(task_img, task_output_filepath)
            
            print(f"Saved task image to: {task_output_filepath}")
            
            # Save volume mapping info
            mapping_file = os.path.join(sub_dir_path, f"sub-{sub}_task_volume_mapping.txt")
            with open(mapping_file, "w") as f:
                f.write("original_volume\ttask_volume\n")
                for orig_vol, task_vol in zip(original_volumes, task_volumes):
                    f.write(f"{orig_vol}\t{task_vol}\n")
    
def extract_fmri_task_data(data_dir, task="dt", space="MNI152NLin2009cAsym",
                    t_r=2.0, func_desc="preproc",
                    condition="create", func_suffix="cleaned"):
    """
    Extract fMRI task data from BIDS dataset and save it.
    """

    print("Starting fMRI task data extraction...")
    print(f"Data directory: {data_dir}")
    print(f"Task: {task}, Space: {space}")

    # BIDS layout
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    subjects = layout.get_subjects()

    print(f"Found {len(subjects)} subjects in the dataset.")

    t_r_value = get_tr_value(data_dir, space=space, task=task, func_desc=func_desc)
    if t_r_value is not None:
        t_r = t_r_value
    else:
        print("No TR value found in metadata, using provided t_r value:", t_r)
    print(f"Using TR value: {t_r} seconds")

    # Parallel processing of subjects
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(partial(extract_fmri_task_data_for_subject,
            data_dir=data_dir, task=task, space=space, t_r=t_r,
            func_desc=func_desc, condition=condition, func_suffix=func_suffix), subjects),
            total=len(subjects), desc="Processing subjects (parallel)"))

def main():
    parser = argparse.ArgumentParser(description="Extract fMRI task data from BIDS dataset.")

    parser.add_argument("-d", "--data-dir", type=str, help="Path to fMRI prep data directory.", required=True)
    parser.add_argument("-t", "--task", type=str, default="dt", help="Task name for fMRI data.")
    parser.add_argument("-s", "--space", type=str, default="MNI152NLin2009cAsym", help="Space for fMRI data.")
    parser.add_argument("--tr", type=float, default=2.0, help="Repetition time (TR) in seconds.")
    parser.add_argument("--func-desc", type=str, default="preproc", help="Description for functional images.")
    parser.add_argument("--condition", type=str, default="create", help="Condition type to extract volumes for.")
    parser.add_argument("--func-suffix", type=str, default="cleaned", help="Suffix for functional images.")

    args = parser.parse_args()

    extract_fmri_task_data(args.data_dir, 
                         task=args.task, space=args.space,
                         func_desc=args.func_desc,
                         condition=args.condition,
                         t_r=args.tr,
                         func_suffix=args.func_suffix)

    print("fMRI task data extraction completed successfully.")

if __name__ == "__main__":
    main()