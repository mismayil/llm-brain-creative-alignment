import argparse
import math
import random
from typing import Optional
from scipy.stats import pearsonr
import pathlib
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from cadabra.utils import read_json, write_json, generate_datetime_id, compute_rsa, get_stimuli_id, get_subject_id
from cadabra.brain.fmri_utils import load_brain_neural_data, filter_dim_mismatch_samples

NOISE_CEILING_TYPES = ["per_voxel", "mean_voxel", "median_voxel", "rsa", "rsa_per_subject", 
                       "mean_voxel_per_subject", "median_voxel_per_subject"]

@dataclass
class NoiseCeilingResult:
    noise_ceiling_values: list
    subjects: Optional[list] = None

def get_stimulus_data(sample, ns_type: str = "per_voxel", voxel: Optional[int] = None):
    if ns_type == "per_voxel" and voxel is not None:
        return sample["fmri_data"][voxel]
    elif "mean_voxel" in ns_type:
        return np.mean(sample["fmri_data"])
    elif "median_voxel" in ns_type:
        return np.median(sample["fmri_data"])
    elif "rsa" in ns_type:
        return sample["fmri_data"]
    else:
        raise ValueError(f"Unknown ns_type: {ns_type}")

def compute_single_trial_corr(brain_data, ns_type: str = "per_voxel", voxel: Optional[int] = None):
    # shuffle samples
    brain_data_shuffled = brain_data.copy()
    random.shuffle(brain_data_shuffled)

    # create two groups
    mid_index = len(brain_data_shuffled) // 2
    group1 = brain_data_shuffled[:mid_index]
    group2 = brain_data_shuffled[mid_index:]

    # find common stimuli between two groups
    stimuli_group1 = set(get_stimuli_id(sample) for sample in group1)
    stimuli_group2 = set(get_stimuli_id(sample) for sample in group2)
    common_stimuli = stimuli_group1.intersection(stimuli_group2)
    if not common_stimuli:
        return 0.0
    
    # collect voxel responses for common stimuli
    group1_data = []
    group2_data = []
    for stimulus in common_stimuli:
        group1_stimulus_data = []
        group2_stimulus_data = []
        
        for sample in group1:
            if get_stimuli_id(sample) == stimulus:
                group1_stimulus_data.append(get_stimulus_data(sample, ns_type, voxel))
        
        for sample in group2:
            if get_stimuli_id(sample) == stimulus:
                group2_stimulus_data.append(get_stimulus_data(sample, ns_type, voxel))
        
        if group1_stimulus_data and group2_stimulus_data:
            mean_group1 = np.stack(group1_stimulus_data).mean(axis=0)
            mean_group2 = np.stack(group2_stimulus_data).mean(axis=0)
            group1_data.append(mean_group1)
            group2_data.append(mean_group2)

    if len(group1_data) < 2 or len(group2_data) < 2:
        return 0.0
    
    # compute correlation
    if ns_type == "rsa":
        group1_data = np.stack(group1_data)
        group2_data = np.stack(group2_data)
        corr, _ = compute_rsa(group1_data, group2_data)
    else:
        corr, _ = pearsonr(group1_data, group2_data)
    return corr

def compute_multi_trial_noise_ceiling(brain_data, ns_type: str = "per_voxel", voxel: Optional[int] = None, num_trials=10):
    trial_corrs = []

    for _ in range(num_trials):
        corr = compute_single_trial_corr(brain_data, ns_type=ns_type, voxel=voxel)
        trial_corrs.append(corr)
    
    noise_ceil = sum(trial_corrs) / num_trials

    # Apply Spearman-Brown correction
    if noise_ceil >= 1.0:
        corrected_noise_ceil = 1.0
    else:
        corrected_noise_ceil = (2 * noise_ceil) / (1 + noise_ceil)
    
    # Return the square root of the corrected noise ceiling
    # because correlation is squared in variance explanation
    corrected_noise_ceil = max(0.0, corrected_noise_ceil)
    corrected_noise_ceil = math.sqrt(corrected_noise_ceil)
    return corrected_noise_ceil

def compute_per_subject_noise_ceiling(brain_data, ns_type: str = "rsa_per_subject"):
    subjects = list(set([get_subject_id(bs) for bs in brain_data]))
    print(f"Found {len(subjects)} subjects in brain data.")
    noise_ceiling_values = []

    with tqdm(desc=f"Computing {ns_type} noise ceiling", total=len(subjects)) as pbar:
        for subject in subjects:
            subject_brain_data = [bs for bs in brain_data if get_subject_id(bs) == subject]
            rest_brain_data = [bs for bs in brain_data if get_subject_id(bs) != subject]
            # find common stimuli between two groups
            subject_stimuli = set(get_stimuli_id(sample) for sample in subject_brain_data)
            rest_stimuli = set(get_stimuli_id(sample) for sample in rest_brain_data)
            common_stimuli = subject_stimuli.intersection(rest_stimuli)

            subject_data = []
            rest_data = []

            for stimulus in common_stimuli:
                subject_stimulus_data = []
                rest_stimulus_data = []
                
                for sample in subject_brain_data:
                    if get_stimuli_id(sample) == stimulus:
                        subject_stimulus_data.append(get_stimulus_data(sample, ns_type))
                
                for sample in rest_brain_data:
                    if get_stimuli_id(sample) == stimulus:
                        rest_stimulus_data.append(get_stimulus_data(sample, ns_type))
                
                if subject_stimulus_data and rest_stimulus_data:
                    mean_subject = np.stack(subject_stimulus_data).mean(axis=0)
                    mean_rest = np.stack(rest_stimulus_data).mean(axis=0)
                    subject_data.append(mean_subject)
                    rest_data.append(mean_rest)
            
            subject_data = np.stack(subject_data)
            rest_data = np.stack(rest_data)

            if ns_type == "rsa_per_subject":
                rsa, _ = compute_rsa(subject_data, rest_data)
                noise_ceiling_values.append(rsa)
            else:
                corr, _ = pearsonr(subject_data.flatten(), rest_data.flatten())
                noise_ceiling_values.append(corr)
            
            pbar.set_description(f"Computing {ns_type} noise ceiling | subject={subject} | ns={noise_ceiling_values[-1]:.4f}")
            pbar.update(1)
            
    return NoiseCeilingResult(noise_ceiling_values=noise_ceiling_values, subjects=subjects)

def compute_noise_ceiling(brain_data_path, type="per_voxel", num_trials=10):
    input_data = read_json(brain_data_path)
    brain_data = input_data["data"]
    
    # Load and flatten fMRI data for each sample
    for sample in tqdm(brain_data, desc="Loading fMRI data", total=len(brain_data)):
        fmri_data = load_brain_neural_data(brain_data_path, sample["fmri_path"])
        fmri_data = np.mean(fmri_data, axis=0)  # average across time if multiple trials exist
        fmri_data = fmri_data.flatten()
        sample["fmri_data"] = fmri_data
    
    # Remove samples that have mismatching fMRI data lengths
    brain_data = filter_dim_mismatch_samples(brain_data, data_field="fmri_data")
    num_voxels = brain_data[0]["fmri_data"].shape[0]

    # Compute noise ceiling based on the specified type
    if type == "per_voxel":
        per_voxel_noise_ceilings = []
        with tqdm(desc="Computing per-voxel noise ceiling", total=num_voxels) as pbar:
            for voxel in range(num_voxels):
                voxel_noise_ceiling = compute_multi_trial_noise_ceiling(brain_data, ns_type="per_voxel", voxel=voxel, num_trials=num_trials)
                per_voxel_noise_ceilings.append(voxel_noise_ceiling)
                min_ceiling = min(per_voxel_noise_ceilings)
                max_ceiling = max(per_voxel_noise_ceilings)
                pbar.set_description(f"Computing per-voxel noise ceiling | min={min_ceiling:.4f}, max={max_ceiling:.4f}")
                pbar.update(1)
        return NoiseCeilingResult(noise_ceiling_values=per_voxel_noise_ceilings)
    elif type == "mean_voxel":
        print("Computing mean-voxel noise ceiling...")
        mean_voxel_noise_ceiling = compute_multi_trial_noise_ceiling(brain_data, ns_type="mean_voxel", num_trials=num_trials)
        print(f"Mean-voxel noise ceiling: {mean_voxel_noise_ceiling:.4f}")
        return NoiseCeilingResult(noise_ceiling_values=[mean_voxel_noise_ceiling])
    elif type == "median_voxel":
        print("Computing median-voxel noise ceiling...")
        median_voxel_noise_ceiling = compute_multi_trial_noise_ceiling(brain_data, ns_type="median_voxel", num_trials=num_trials)
        print(f"Median-voxel noise ceiling: {median_voxel_noise_ceiling:.4f}")
        return NoiseCeilingResult(noise_ceiling_values=[median_voxel_noise_ceiling])
    elif type == "rsa":
        print("Computing RSA noise ceiling...")
        rsa_noise_ceiling = compute_multi_trial_noise_ceiling(brain_data, ns_type="rsa", num_trials=num_trials)
        print(f"RSA noise ceiling: {rsa_noise_ceiling:.4f}")
        return NoiseCeilingResult(noise_ceiling_values=[rsa_noise_ceiling])
    elif "per_subject" in type:
        print("Computing per-subject noise ceiling...")
        per_subject_noise_ceiling = compute_per_subject_noise_ceiling(brain_data, ns_type=type)
        return per_subject_noise_ceiling
    else:
        raise ValueError(f"Unknown noise ceiling computation type: {type}")

def main():
    parser = argparse.ArgumentParser(description="Compute noise ceiling for brain data.")
    parser.add_argument("-d", "--data-path", type=str, required=True, help="Path to the prepared brain data.")
    parser.add_argument("--type", type=str, default="per_voxel", choices=NOISE_CEILING_TYPES, help="Type of noise ceiling computation.")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials for estimation.")

    args = parser.parse_args()

    print(f"Computing noise ceiling for data in {args.data_path}...")

    noise_ceiling_results = compute_noise_ceiling(
        brain_data_path=args.data_path,
        type=args.type,
        num_trials=args.num_trials
    )

    noise_ceiling_data = noise_ceiling_results.noise_ceiling_values
    subjects = noise_ceiling_results.subjects

    run_id = generate_datetime_id()
    output_dir = pathlib.Path(args.data_path).parent / "noise_ceilings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"noise_ceiling_{args.type}_n{args.num_trials}_{run_id}.json"
    data_output_path = output_path.with_name(output_path.stem + f"_data.npy")

    np.save(data_output_path, np.array(noise_ceiling_data))

    output_data = {
        "metadata": {
            "output_path": str(output_path),
            "data_path": args.data_path,
            "noise_ceiling_type": args.type,
            "num_trials": args.num_trials,
            "subjects": subjects
        },
        "metrics": {
            "min_noise_ceiling": float(np.nanmin(noise_ceiling_data)),
            "max_noise_ceiling": float(np.nanmax(noise_ceiling_data)),
            "mean_noise_ceiling": float(np.nanmean(noise_ceiling_data)),
            "median_noise_ceiling": float(np.nanmedian(noise_ceiling_data))
        },
        "data": {
            "noise_ceiling_path": data_output_path.name
        }
    }

    write_json(output_data, output_path)
    print(f"Noise ceiling computed and saved to {output_path}.")

if __name__ == "__main__":
    main()