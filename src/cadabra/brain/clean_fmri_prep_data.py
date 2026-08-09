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

from cadabra.brain.fmri_utils import get_tr_value, DEFAULT_CONFOUND_VARS

def save_clean_fmri_prep_data_for_subject(sub, data_dir, task="dt", space="MNI152NLin2009cAsym",
                                          high_pass=0.009, low_pass=None, t_r=2.0,
                                          confound_vars=DEFAULT_CONFOUND_VARS,
                                          func_desc="preproc", mask_desc="brain", confound_desc="confounds",
                                          detrend=True, standardize=True):
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    func_files = layout.get(subject=sub, datatype="func", task=task, desc=func_desc,
                            space=space, extension="nii.gz", suffix="bold", return_type="file")
    mask_files = layout.get(subject=sub, datatype="func", task=task, desc=mask_desc, suffix="mask",
                            space=space, extension="nii.gz", return_type="file")
    confound_files = layout.get(subject=sub, datatype="func", task=task, desc=confound_desc,
                                extension="tsv", return_type="file")

    if not (len(func_files) == len(mask_files) == len(confound_files)):
        print(f"Warning: Mismatch in number of functional, mask, and confound files for sub-{sub}. Skipping this subject.")
        print(f"Functional files: {len(func_files)}, Mask files: {len(mask_files)}, Confound files: {len(confound_files)}")
        return

    sub_dir_path = os.path.join(data_dir, f"sub-{sub}", "func")

    for func_file, mask_file, confound_file in zip(func_files, mask_files, confound_files):
        original_filename = os.path.basename(func_file)
        cleaned_filename = original_filename.replace(".nii.gz", "_cleaned.nii.gz")
        output_filepath = os.path.join(sub_dir_path, cleaned_filename)

        if os.path.exists(output_filepath):
            print(f"Cleaned file already exists for sub-{sub}: {output_filepath}. Skipping.")
            continue

        confound_df = pd.read_csv(confound_file, delimiter="\t")

        derivative_columns = [f"{c}_derivative1" for c in confound_vars]
        final_confounds = confound_vars + derivative_columns

        confound_df = confound_df[final_confounds]

        # Load the full 4D functional image
        func_img = nimg.load_img(func_file)

        # Confirm dimensions match
        if func_img.shape[-1] != confound_df.shape[0]:
            raise ValueError(f"TR mismatch for sub-{sub}: {func_img.shape[-1]} volumes vs {confound_df.shape[0]} confound timepoints.")

        # Check if there are any NaN values in the confound DataFrame
        nan_rows = confound_df[confound_df.isna().any(axis=1)]
        nan_row_indices = nan_rows.index.tolist()

        # If there are NaN values, drop those rows from the confound DataFrame and the functional image
        if nan_row_indices:
            print(f"Warning: Found NaN values in confounds for sub-{sub}, dropping {len(nan_row_indices)} rows.")
            confound_df = confound_df.dropna()
            func_data = func_img.get_fdata()
            func_data = np.delete(func_data, nan_row_indices, axis=3)
            func_img = nib.Nifti1Image(func_data, func_img.affine, func_img.header)
            # save nan row indices to a text file
            nan_indices_file = os.path.join(sub_dir_path, f"sub-{sub}_nan_indices.txt")
            with open(nan_indices_file, "w") as f:
                f.write("\n".join(map(str, nan_row_indices)))

        confounds_matrix = confound_df.values

        # Clean the image
        clean_img = nimg.clean_img(func_img,
                                confounds=confounds_matrix,
                                detrend=detrend,
                                standardize=standardize,
                                low_pass=low_pass,
                                high_pass=high_pass,
                                t_r=t_r,
                                mask_img=mask_file)

        print(f"Saved cleaned image for sub-{sub} to {output_filepath}")
        clean_img.to_filename(output_filepath)
    
def save_clean_fmri_prep_data(data_dir, task="dt", space="MNI152NLin2009cAsym",
                    high_pass=0.009, low_pass=None, t_r=2.0,
                    confound_vars=DEFAULT_CONFOUND_VARS,
                    func_desc="preproc",
                    mask_desc="brain",
                    confound_desc="confounds_regressors",
                    detrend=True,
                    standardize=True):
    """
    Clean fMRI preprocessed data by removing confounds and saving cleaned images.
    """

    print("Starting fMRI prep data cleaning...")
    print(f"Data directory: {data_dir}")
    print(f"Task: {task}, Space: {space}")

    print(f"High-pass filter cutoff: {high_pass} Hz")
    if low_pass is not None:
        print(f"Low-pass filter cutoff: {low_pass} Hz")
    else:
        print("No low-pass filter applied.")
    print(f"Confound variables: {confound_vars}")
    print(f"Detrend: {'Enabled' if detrend else 'Disabled'}")
    print(f"Standardize: {'Enabled' if standardize else 'Disabled'}")

    # BIDS layout
    layout = bids.BIDSLayout(data_dir, validate=False, config=["bids", "derivatives"])
    subjects = layout.get_subjects()

    print(f"Found {len(subjects)} subjects in the dataset.")

    t_r_value = get_tr_value(data_dir, space=space, task=task, func_desc=func_desc)
    if t_r_value is not None:
        print(f"Using TR value: {t_r_value} seconds")
        t_r = t_r_value
    else:
        print("No TR value found in metadata, using provided t_r value:", t_r)
    
    # Parallel processing of subjects
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(partial(save_clean_fmri_prep_data_for_subject,
            data_dir=data_dir, task=task, space=space,
            high_pass=high_pass, low_pass=low_pass, t_r=t_r,
            confound_vars=confound_vars,
            func_desc=func_desc, mask_desc=mask_desc, confound_desc=confound_desc,
            detrend=detrend, standardize=standardize), subjects),
            total=len(subjects), desc="Processing subjects (parallel)"))

def main():
    parser = argparse.ArgumentParser(description="Clean fMRI preprocessed data by removing confounds.")

    parser.add_argument("-d", "--data-dir", type=str, help="Path to fMRI prep data directory.", required=True)
    parser.add_argument("-t", "--task", type=str, default="dt", help="Task name for fMRI data.")
    parser.add_argument("-s", "--space", type=str, default="MNI152NLin2009cAsym", help="Space for fMRI data.")
    parser.add_argument("--high-pass", type=float, default=0.009, help="High-pass filter cutoff frequency.")
    parser.add_argument("--low-pass", type=float, default=None, help="Low-pass filter cutoff frequency.")
    parser.add_argument("--tr", type=float, default=2.0, help="Repetition time (TR) in seconds.")
    parser.add_argument("--confound-vars", type=str, nargs="*", default=DEFAULT_CONFOUND_VARS,
                        help="List of confound variables to include in cleaning.")
    parser.add_argument("--no-detrend", action="store_true", help="Whether not to detrend the data.")
    parser.add_argument("--no-standardize", action="store_true", help="Whether not to standardize the data.")
    parser.add_argument("--func-desc", type=str, default="preproc", help="Description for functional images.")
    parser.add_argument("--mask-desc", type=str, default="brain", help="Description for brain mask images.")
    parser.add_argument("--confound-desc", type=str, default="confounds", help="Description for confound files.")

    args = parser.parse_args()

    save_clean_fmri_prep_data(args.data_dir, 
                         task=args.task, space=args.space,
                         high_pass=args.high_pass, 
                         low_pass=args.low_pass, t_r=args.tr,
                         confound_vars=args.confound_vars,
                         func_desc=args.func_desc,
                         mask_desc=args.mask_desc,
                         confound_desc=args.confound_desc,
                         detrend=not args.no_detrend,
                         standardize=not args.no_standardize)

    print("fMRI prep data cleaning completed successfully.")

if __name__ == "__main__":
    main()