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
from nilearn.glm.first_level import FirstLevelModel

from cadabra.brain.fmri_utils import get_tr_value, DEFAULT_CONFOUND_VARS
from cadabra.utils import write_json, generate_datetime_id

def compute_contrast_name(contrast_def, treatment_task, control_task):
    if "treatment" in contrast_def and "control" in contrast_def:
        return contrast_def.replace("treatment", treatment_task).replace("control", control_task)
    elif "treatment" in contrast_def:
        return contrast_def.replace("treatment", treatment_task)
    elif "control" in contrast_def:
        return contrast_def.replace("control", control_task)
    else:
        raise ValueError(f"Invalid contrast definition: {contrast_def}")

def compute_and_save_contrast(contrast_name, first_level_model, sub, sub_dir_path, original_filename, output_type, output_name, result):
    print(f"Computing contrast '{contrast_name}' with output type '{output_type}' for sub-{sub}...")
    contrast_img = first_level_model.compute_contrast(
        contrast_name,
        output_type=output_type  # z-score or 'stat' for t-map, 'effect_size' for beta difference
    )
    contrast_filename = original_filename.replace(".nii.gz", f"_flm_{contrast_name}_{output_name}.nii.gz")
    contrast_output_filepath = os.path.join(sub_dir_path, contrast_filename)
    contrast_img.to_filename(contrast_output_filepath)
    print(f"Saved contrast image for sub-{sub} to {contrast_output_filepath}")
    return contrast_output_filepath

def run_first_level_glm_for_subject(sub, data_dir, output_dir, task="dt", space="MNI152NLin2009cAsym",
                                    hrf_model="glover", drift_model="cosine", noise_model="ar1",
                                    high_pass=0.009, t_r=2.0,
                                    confound_vars=DEFAULT_CONFOUND_VARS,
                                    func_desc="preproc", confound_desc="confounds",
                                    standardize=True, treatment_task="create", control_task="object", 
                                    contrast_def="treatment-control",
                                    compute_per_stimulus=False):
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    func_files = layout.get(subject=sub, datatype="func", task=task, desc=func_desc,
                            space=space, extension="nii.gz", suffix="bold", return_type="file")
    confound_files = layout.get(subject=sub, datatype="func", task=task, desc=confound_desc,
                                extension="tsv", return_type="file")
    events_files = layout.get(subject=sub, datatype="func", task=task, suffix="events",
                                extension="tsv", return_type="file")

    if not (len(func_files) == len(confound_files) == len(events_files)):
        print(f"Warning: Mismatch in number of functional, confound, and events files for sub-{sub}. Skipping this subject.")
        print(f"Functional files: {len(func_files)}, Confound files: {len(confound_files)}, Events files: {len(events_files)}")
        return

    sub_dir_path = os.path.join(output_dir, f"sub-{sub}", "func")
    os.makedirs(sub_dir_path, exist_ok=True)

    func_file = func_files[0]
    confounds_file = confound_files[0]
    events_file = events_files[0]

    original_filename = os.path.basename(func_file)

    func_img = nimg.load_img(func_file)
    confound_df = pd.read_csv(confounds_file, delimiter="\t")
    events_df = pd.read_csv(events_file, delimiter="\t")

    derivative_columns = [f"{c}_derivative1" for c in DEFAULT_CONFOUND_VARS]
    final_confounds = confound_vars + derivative_columns

    confound_df = confound_df[final_confounds]
    confound_df = confound_df.bfill()

    # --- Build the model ---
    first_level_model = FirstLevelModel(
        t_r=t_r,                          # repetition time in seconds (update to your TR)
        hrf_model=hrf_model,              # canonical HRF (can also use 'spm' or 'fir')
        drift_model=drift_model,          # default high-pass filter
        high_pass=high_pass,              # Hz, typical
        noise_model=noise_model,          # autoregressive noise
        standardize=standardize,          # z-score signals
        minimize_memory=False,
    )

    if compute_per_stimulus:
        if "treatment" in contrast_def and "control" in contrast_def:
            raise ValueError("Cannot compute per-stimulus contrasts for combined contrasts like 'treatment-control'. Please choose either 'treatment' or 'control'.")
    
        # Create separate conditions for each stimulus
        new_events = []
        stim_i = 1
        for _, row in events_df.iterrows():
            new_event = row.copy()
            if contrast_def == "treatment" and row['trial_type'] == treatment_task or \
               contrast_def == "control" and row['trial_type'] == control_task:
                new_event['trial_type'] = f"{row['trial_type']}_stim{stim_i}"
                stim_i += 1
            new_events.append(new_event)
        events_df = pd.DataFrame(new_events)

    # --- Fit the model ---
    first_level_model = first_level_model.fit(
        func_img,
        events=events_df,
        confounds=confound_df,
    )

    result = {
        "subject": sub,
    }

    # Compute the contrast
    for output_type, output_name in [("z_score", "z_map"), ("effect_size", "beta_map")]:
        contrast_name = compute_contrast_name(contrast_def, treatment_task, control_task)
        if compute_per_stimulus:
            stimulus_conditions = events_df['trial_type'].unique()
            stimulus_conditions = [cond for cond in stimulus_conditions if (contrast_def == "treatment" and cond.startswith(treatment_task)) or (contrast_def == "control" and cond.startswith(control_task))]
            stimulus_contrast_files = []
            for stim_cond in stimulus_conditions:
                stim_contrast_name = stim_cond
                contrast_filepath = compute_and_save_contrast(
                    stim_contrast_name, first_level_model, sub, sub_dir_path,
                    original_filename, output_type, output_name, result
                )
                stimulus_contrast_files.append(contrast_filepath)
            result[f"contrast_{contrast_name}_{output_name}"] = stimulus_contrast_files
        else:
            contrast_filepath = compute_and_save_contrast(
                contrast_name, first_level_model, sub, sub_dir_path,
                original_filename, output_type, output_name, result
            )
            result[f"contrast_{contrast_name}_{output_name}"] = contrast_filepath

    return result

def run_first_level_glm(data_dir, output_dir, task="dt", space="MNI152NLin2009cAsym",
                        hrf_model="glover", drift_model="cosine", noise_model="ar1",
                        high_pass=0.009, t_r=2.0,
                        confound_vars=DEFAULT_CONFOUND_VARS,
                        func_desc="preproc", confound_desc="confounds",
                        standardize=True, treatment_task="create", control_task="object",
                        exclude_subjects=None, parallel=False, num_workers=None, 
                        contrast_def="treatment-control", compute_per_stimulus=False):
    """
    Run first-level GLM analysis for all subjects in the BIDS dataset.
    """

    print("Running first-level GLM analysis with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Task: {task}, Space: {space}")

    print(f"High-pass filter cutoff: {high_pass} Hz")
    print(f"Confound variables: {confound_vars}")
    print(f"Standardize: {'Enabled' if standardize else 'Disabled'}")
    print(f"HRF model: {hrf_model}, Drift model: {drift_model}, Noise model: {noise_model}")
    print(f"Treatment task: {treatment_task}, Control task: {control_task}")

    # BIDS layout
    print("Loading BIDS dataset...")
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    subjects = layout.get_subjects()

    print(f"Found {len(subjects)} subjects in the dataset.")

    if exclude_subjects:
        subjects = [sub for sub in subjects if sub not in exclude_subjects]
        print(f"Excluding subjects: {exclude_subjects}")
        print(f"{len(subjects)} subjects remaining after exclusion.")

    t_r_value = get_tr_value(data_dir, space=space, task=task, func_desc=func_desc)
    if t_r_value is not None:
        print(f"Using TR value: {t_r_value} seconds")
        t_r = t_r_value
    else:
        print("No TR value found in metadata, using provided t_r value:", t_r)
    
    if not parallel:
        results = []
        for sub in tqdm(subjects, desc="Processing subjects (sequential)"):
            result = run_first_level_glm_for_subject(
                sub, data_dir, output_dir, task=task, space=space,
                hrf_model=hrf_model, drift_model=drift_model, noise_model=noise_model,
                high_pass=high_pass, t_r=t_r,
                confound_vars=confound_vars,
                func_desc=func_desc, confound_desc=confound_desc,
                standardize=standardize, treatment_task=treatment_task, control_task=control_task,
                contrast_def=contrast_def,
                compute_per_stimulus=compute_per_stimulus
            )
            results.append(result)
        return results

    num_workers = num_workers or os.cpu_count() or 1
    print(f"Running in parallel using {num_workers} workers...")

    # Parallel processing of subjects
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(partial(run_first_level_glm_for_subject,
            data_dir=data_dir, output_dir=output_dir, task=task, space=space,
            hrf_model=hrf_model, drift_model=drift_model, noise_model=noise_model,
            high_pass=high_pass, t_r=t_r,
            confound_vars=confound_vars,
            func_desc=func_desc, confound_desc=confound_desc,
            standardize=standardize, treatment_task=treatment_task, control_task=control_task,
            contrast_def=contrast_def,
            compute_per_stimulus=compute_per_stimulus), subjects),
            total=len(subjects), desc="Processing subjects (parallel)"))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run first-level GLM analysis on fMRI data.")

    parser.add_argument("-d", "--data-dir", type=str, help="Path to fMRI prep data directory.", required=True)
    parser.add_argument("-o", "--output-dir", type=str, help="Path to output directory for GLM results.", required=True)
    parser.add_argument("-t", "--task", type=str, default="dt", help="Task name for fMRI data.")
    parser.add_argument("-s", "--space", type=str, default="MNI152NLin2009cAsym", help="Space for fMRI data.")
    parser.add_argument("--contrast-def", type=str, default="treatment-control", choices=["treatment", "control", "treatment-control"], help="Name of the contrast to compute.")
    parser.add_argument("--compute-per-stimulus", action="store_true", help="Whether to compute contrasts per stimulus.")
    parser.add_argument("--high-pass", type=float, default=0.009, help="High-pass filter cutoff frequency.")
    parser.add_argument("--tr", type=float, default=2.0, help="Repetition time (TR) in seconds.")
    parser.add_argument("--confound-vars", type=str, nargs="*", default=DEFAULT_CONFOUND_VARS,
                        help="List of confound variables to include in cleaning.")
    parser.add_argument("--no-standardize", action="store_true", help="Whether not to standardize the data.")
    parser.add_argument("--func-desc", type=str, default="preproc", help="Description for functional images.")
    parser.add_argument("--confound-desc", type=str, default="confounds", help="Description for confound files.")
    parser.add_argument("--treatment-task", type=str, default="create", help="Name of the treatment task in events file.")
    parser.add_argument("--control-task", type=str, default="object", help="Name of the control task in events file.")
    parser.add_argument("--hrf-model", type=str, default="spm", help="HRF model to use.")
    parser.add_argument("--drift-model", type=str, default="cosine", help="Drift model to use.")
    parser.add_argument("--noise-model", type=str, default="ar1", help="Noise model to use.")
    parser.add_argument("--exclude-subjects", type=str, nargs='*', default=[], help="List of subject IDs to exclude from processing.")
    parser.add_argument("--parallel", action="store_true", help="Whether to run processing in parallel.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers to use if --parallel is set.")

    args = parser.parse_args()

    run_id = generate_datetime_id()
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    results = run_first_level_glm(args.data_dir, output_dir,
                         task=args.task, space=args.space,
                         high_pass=args.high_pass, 
                         t_r=args.tr,
                         confound_vars=args.confound_vars,
                         func_desc=args.func_desc,
                         confound_desc=args.confound_desc,
                         standardize=not args.no_standardize,
                         treatment_task=args.treatment_task,
                         control_task=args.control_task,
                         hrf_model=args.hrf_model,
                         drift_model=args.drift_model,
                         noise_model=args.noise_model,
                         exclude_subjects=args.exclude_subjects,
                         parallel=args.parallel,
                         num_workers=args.num_workers,
                         contrast_def=args.contrast_def,
                         compute_per_stimulus=args.compute_per_stimulus)

    output_path = os.path.join(output_dir, f"first_level_glm_results_{run_id}.json")

    output_data = {
        "metadata": {
            "output_path": output_path,
            "run_id": run_id,
            "data_dir": args.data_dir,
            "output_dir": output_dir,
            "task": args.task,
            "space": args.space,
            "high_pass": args.high_pass,
            "t_r": args.tr,
            "confound_vars": args.confound_vars,
            "func_desc": args.func_desc,
            "confound_desc": args.confound_desc,
            "standardize": not args.no_standardize,
            "treatment_task": args.treatment_task,
            "control_task": args.control_task,
            "hrf_model": args.hrf_model,
            "drift_model": args.drift_model,
            "noise_model": args.noise_model,
            "exclude_subjects": args.exclude_subjects,
            "parallel": args.parallel,
            "num_workers": args.num_workers,
            "contrast_def": args.contrast_def,
            "compute_per_stimulus": args.compute_per_stimulus
        },
        "data": results 
    }

    write_json(output_data, output_path)

    print("First-level GLM analysis completed successfully.")

if __name__ == "__main__":
    main()