import argparse
import os
import pandas as pd
import bids
from tqdm import tqdm
from nilearn import image as nimg
import numpy as np
import math

from cadabra.utils import read_json, write_json, convert_nan_to_none, generate_hash_id, find_files, json_serialize, generate_datetime_id
from cadabra.brain.fmri_utils import get_tr_value, load_roi_mask, apply_roi_mask, filter_dim_mismatch_samples

def prepare_templeton_aut_brain_data(datapath, metadata_path, task="dt", space="MNI152NLin2009cAsym",
                        tr=2.0, func_desc="preproc", condition='create', 
                        roi="whole_brain", roi_resample=False, roi_interpolation="nearest",
                        roi_path=None, roi_threshold=0.0, exclude_subjects=None, 
                        **kwargs):
    """
    Prepare final human brain fMRI data for benchmarking by extracting task-specific volumes.
    """
    print("Starting preparation of human brain fMRI data...")
    print(f"Data directory: {datapath}")
    print(f"Metadata path: {metadata_path}")
    print(f"Task: {task}, Space: {space}")
    print(f"Functional description: {func_desc}")
    print(f"Condition: {condition}")
    print(f"Region of interest (ROI): {roi}")
    print(f"Repetition time (TR): {tr} seconds")
    print(f"ROI resample: {roi_resample}, Interpolation: {roi_interpolation}")
    if roi_path:
        print(f"Custom ROI path: {roi_path}")
    print(f"ROI threshold: {roi_threshold}")

    # Load item metadata
    if metadata_path.endswith('.csv'):
        metadata = pd.read_csv(metadata_path)
    elif metadata_path.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_path)
    else:
        raise ValueError("Item path must point to a .csv or .xlsx file.")

    print("Preparing BIDS layout...")
    # BIDS layout
    layout = bids.BIDSLayout(datapath, validate=False, config=["bids", "derivatives"])
    subjects = layout.get_subjects()

    print(f"Found {len(subjects)} subjects in the dataset.")

    if exclude_subjects:
        subjects = [sub for sub in subjects if sub not in exclude_subjects]
        print(f"Excluding subjects: {exclude_subjects}. Remaining subjects: {len(subjects)}")

    print("Loading ROI mask...")
    roi_mask = load_roi_mask(roi, roi_path=roi_path, roi_threshold=roi_threshold)

    brain_data = []

    for sub in tqdm(subjects, desc="Processing subjects"):
        func_files = layout.get(subject=sub, datatype="func", task=task, desc=func_desc,
                                space=space, extension="nii.gz", suffix=condition, return_type="file")
        events_files = layout.get(subject=sub, datatype="func", task=task, suffix="events",
                                    extension="tsv", return_type="file")

        if not (len(func_files) == len(events_files)):
            print(f"Warning: Mismatch in number of functional, and events files for sub-{sub}. Skipping this subject.")
            continue

        sub_metadata = metadata[(metadata['id'] == int(sub)) & (metadata['condition'] == condition)]

        for func_file, events_file in zip(func_files, events_files):
            func_img = nimg.load_img(func_file)   # shape (V_x, V_y, V_z, T)
            func_data = apply_roi_mask(func_img, roi_mask, resample=roi_resample, interpolation=roi_interpolation)  # shape (T, V) where V is number of voxels in ROI
            events = pd.read_csv(events_file, sep="\t")
            task_events = events[events['trial_type'] == condition]
            task_events_iter = task_events.iterrows()
            func_data_lst = [func_data[i] for i in range(func_data.shape[0])]
            func_data_iter = iter(func_data_lst)

            for i, item_info in sub_metadata.iterrows():
                try:
                    _, event = next(task_events_iter)
                except StopIteration:
                    print(f"Warning: Not enough events for subject {sub}.")
                    break

                duration = event['duration']
                num_volumes = math.ceil(duration / tr)

                item_func_data = []

                for _ in range(num_volumes):
                    try:
                        vol_data = next(func_data_iter)
                        item_func_data.append(vol_data)
                    except StopIteration:
                        print(f"Warning: Not enough volumes for subject {sub}.")
                        break
                item_func_data = np.stack(item_func_data)
                stimuli_id = item_info["stimuli"].replace(" ", "_")
                ratings = [item_info["rater1"], item_info["rater2"], item_info["rater3"], item_info["rater4"]]
                mean_rating = np.nanmean(ratings)
                brain_data.append({
                    "id": generate_hash_id(f"sub-{sub}_item-{stimuli_id}_condition-{condition}_roi-{roi.replace(':', '_')}"),
                    "subject_id": sub,
                    "condition": condition,
                    "stimuli_id": stimuli_id,
                    "stimuli": item_info["stimuli"],
                    "response": convert_nan_to_none(item_info["response"]),
                    "ratings": json_serialize([convert_nan_to_none(r) for r in ratings]),
                    "mean_rating": json_serialize(convert_nan_to_none(mean_rating)),
                    "fmri_data": item_func_data
                })
    
    return brain_data

def prepare_templeton_aut_brain_data_from_flm(datapath, metadata_path, task="dt", space="MNI152NLin2009cAsym",
                        tr=2.0, func_desc="preproc", condition='create', 
                        roi="whole_brain", roi_resample=False, roi_interpolation="nearest",
                        roi_path=None, roi_threshold=0.0, exclude_subjects=None, sub_stimuli_field="contrast_create_beta_map", 
                        **kwargs):
    """
    Prepare final human brain fMRI data for benchmarking by extracting task-specific volumes from first-level GLM beta maps.
    """
    print("Starting preparation of human brain fMRI data...")
    print(f"Data path: {datapath}")
    print(f"Metadata path: {metadata_path}")
    print(f"Task: {task}, Space: {space}")
    print(f"Functional description: {func_desc}")
    print(f"Condition: {condition}")
    print(f"Region of interest (ROI): {roi}")
    print(f"Repetition time (TR): {tr} seconds")
    print(f"ROI resample: {roi_resample}, Interpolation: {roi_interpolation}")
    if roi_path:
        print(f"Custom ROI path: {roi_path}")
    print(f"ROI threshold: {roi_threshold}")
    print(f"Sub stimuli field: {sub_stimuli_field}")

    # Load item metadata
    if metadata_path.endswith('.csv'):
        metadata = pd.read_csv(metadata_path)
    elif metadata_path.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_path)
    else:
        raise ValueError("Item path must point to a .csv or .xlsx file.")

    flm_contents = read_json(datapath)
    flm_data = flm_contents['data']

    print(f"Found {len(flm_data)} subjects in the dataset.")

    if exclude_subjects:
        flm_data = [sub_result for sub_result in flm_data if sub_result["subject"] not in exclude_subjects]
        print(f"Excluding subjects: {exclude_subjects}. Remaining subjects: {len(flm_data)}")

    print("Loading ROI mask...")
    roi_mask = load_roi_mask(roi, roi_path=roi_path, roi_threshold=roi_threshold)

    brain_data = []

    for subject_data in tqdm(flm_data, desc="Processing subjects"):
        sub = subject_data["subject"]
        sub_metadata = metadata[(metadata['id'] == int(sub)) & (metadata['condition'] == condition)]
        sub_stimuli_files = subject_data[sub_stimuli_field]
        sub_stimuli_files = sorted(sub_stimuli_files, key=lambda x: int(x.split("stim")[-1].split("_")[0]))

        for sub_stimuli_file, sub_stimuli_info in zip(sub_stimuli_files, sub_metadata.to_dict('records')):
            stimuli_img = nimg.load_img(sub_stimuli_file)   # shape (V_x, V_y, V_z, T) or (V_x, V_y, V_z)
            # if number of dimensions is 3, expand to 4D at the last dimension
            if len(stimuli_img.shape) <= 3:
                stimuli_img = nimg.concat_imgs([stimuli_img])
            stimuli_img_data = apply_roi_mask(stimuli_img, roi_mask, resample=roi_resample, interpolation=roi_interpolation)  # shape (T, V) where V is number of voxels in ROI
            stimuli_id = sub_stimuli_info["stimuli"].replace(" ", "_")
            ratings = [sub_stimuli_info["rater1"], sub_stimuli_info["rater2"], sub_stimuli_info["rater3"], sub_stimuli_info["rater4"]]
            mean_rating = np.nanmean(ratings)
            brain_data.append({
                "id": generate_hash_id(f"sub-{sub}_item-{stimuli_id}_condition-{condition}_roi-{roi.replace(':', '_')}"),
                "subject_id": sub,
                "condition": condition,
                "stimuli_id": stimuli_id,
                "stimuli": sub_stimuli_info["stimuli"],
                "response": convert_nan_to_none(sub_stimuli_info["response"]),
                "ratings": json_serialize([convert_nan_to_none(r) for r in ratings]),
                "mean_rating": json_serialize(convert_nan_to_none(mean_rating)),
                "fmri_data": stimuli_img_data
            })
    
    return brain_data

def prepare_lebel_prep_brain_data(datapath, roi="whole_brain", roi_resample=False, roi_interpolation="nearest",
                        roi_path=None, roi_threshold=0.0, exclude_subjects=None, **kwargs):
    """
    Prepare final Lebel preprocessed human brain fMRI data from https://www.nature.com/articles/s41597-023-02437-z.
    """
    import h5py
    print("Starting preparation of human brain fMRI data...")
    print(f"Data directory: {datapath}")
    print(f"Region of interest (ROI): {roi}")
    print(f"ROI resample: {roi_resample}, Interpolation: {roi_interpolation}")
    if roi_path:
        print(f"Custom ROI path: {roi_path}")
    print(f"ROI threshold: {roi_threshold}")

    subjects = [d for d in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, d))]

    print(f"Found {len(subjects)} subjects in the dataset.")
    print("Loading ROI mask...")
    roi_mask = load_roi_mask(roi, roi_path=roi_path, roi_threshold=roi_threshold)

    brain_data = []

    if exclude_subjects:
        subjects = [sub for sub in subjects if sub not in exclude_subjects]
        print(f"Excluding subjects: {exclude_subjects}. Remaining subjects: {len(subjects)}")

    for subject in tqdm(subjects, desc="Processing subjects", leave=False, position=0):
        func_files = find_files(os.path.join(datapath, subject), extension="hf5")

        for func_file in tqdm(func_files, desc="Processing functional files", leave=False, position=1):
            filename = os.path.basename(func_file)
            item = filename.split(".")[0]
            with h5py.File(func_file, 'r') as hf:
                data_key = list(hf.keys())[0]
                func_data = np.array(hf[data_key])  # shape (T, V)
                # func_data = apply_roi_mask(func_data, roi_mask, resample=roi_resample, interpolation=roi_interpolation)  # shape (T, V_roi)

            brain_data.append({
                "id": generate_hash_id(f"sub-{subject}_item-{item}_roi-{roi.replace(':', '_')}"),
                "subject_id": subject,
                "stimuli_id": item,
                "fmri_data": func_data
            })
    
    return brain_data

def post_filter_brain_data(brain_data_or_path, exclude_subjects=None, remove_na_responses=False,
                           min_stimuli_rating=None, max_stimuli_rating=None, min_subject_rating=None, max_subject_rating=None,
                           skip_dim_mismatch_filter=False, **kwargs):
    """
    Post-process brain data by applying various filters.
    """
    if isinstance(brain_data_or_path, str):
        brain_data = read_json(brain_data_or_path)
        brain_data = brain_data['data'] if 'data' in brain_data else brain_data
    else:
        brain_data = brain_data_or_path
    
    if exclude_subjects is not None:
        print(f"Excluding subjects: {exclude_subjects}")
        brain_data = [sample for sample in brain_data if sample['subject_id'] not in exclude_subjects]

    if not skip_dim_mismatch_filter:
        print("Filtering samples with dimension mismatch...")
        brain_data = filter_dim_mismatch_samples(brain_data, data_field="fmri_data")
    
    if remove_na_responses:
        print("Removing samples with NaN responses...")
        brain_data = [sample for sample in brain_data if sample['response'] is not None]

    if min_stimuli_rating is not None:
        print(f"Applying minimum stimuli rating filter: {min_stimuli_rating}")
        brain_data = [sample for sample in brain_data if sample.get('mean_rating') is not None and sample['mean_rating'] >= min_stimuli_rating]
    
    if max_stimuli_rating is not None:
        print(f"Applying maximum stimuli rating filter: {max_stimuli_rating}")
        brain_data = [sample for sample in brain_data if sample.get('mean_rating') is not None and sample['mean_rating'] <= max_stimuli_rating]
    
    if min_subject_rating is not None or max_subject_rating is not None:
        print("Applying subject-level rating filters...")
        subject_rating_data = {}
        for sample in brain_data:
            sub = sample['subject_id']
            if sub not in subject_rating_data:
                subject_rating_data[sub] = []
            if sample.get('mean_rating') is not None:
                subject_rating_data[sub].append(sample['mean_rating'])
        if min_subject_rating is not None:
            valid_subjects = {sub for sub, ratings in subject_rating_data.items() if np.mean(ratings) >= min_subject_rating}
            brain_data = [sample for sample in brain_data if sample['subject_id'] in valid_subjects]
        if max_subject_rating is not None:
            valid_subjects = {sub for sub, ratings in subject_rating_data.items() if np.mean(ratings) < max_subject_rating}
            brain_data = [sample for sample in brain_data if sample['subject_id'] in valid_subjects]

    return brain_data

def add_rating_data(datapath, metadata_path, **kwargs):
    """
    Add rating data to brain data samples based on item metadata.
    """
    brain_data = read_json(datapath)
    brain_data = brain_data['data'] if 'data' in brain_data else brain_data

    # Load item metadata
    if metadata_path.endswith('.csv'):
        metadata = pd.read_csv(metadata_path)
    elif metadata_path.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_path)
    else:
        raise ValueError("Item path must point to a .csv or .xlsx file.")

    for sample in tqdm(brain_data, desc="Adding rating data to samples"):
        sub = sample['subject_id']
        stimuli = sample['stimuli']
        item_metadata = metadata[(metadata['id'] == int(sub)) & (metadata['stimuli'] == stimuli)]
        if not item_metadata.empty:
            item_info = item_metadata.iloc[0]
            ratings = [item_info["rater1"], item_info["rater2"], item_info["rater3"], item_info["rater4"]]
            mean_rating = np.nanmean(ratings)
            sample['ratings'] = json_serialize([convert_nan_to_none(r) for r in ratings])
            sample['mean_rating'] = json_serialize(convert_nan_to_none(mean_rating))
        else:
            sample['ratings'] = None
            sample['mean_rating'] = None

    return brain_data

DATASET_MAP = {
    "templeton_aut": prepare_templeton_aut_brain_data,
    "templeton_aut_flm": prepare_templeton_aut_brain_data_from_flm,
    "lebel_prep": prepare_lebel_prep_brain_data,
    "add_rating": add_rating_data,
}

def main():
    parser = argparse.ArgumentParser(description="Prepare final human brain fMRI data for benchmarking.")

    parser.add_argument("-d", "--datapath", type=str, help="Path to fMRI prep data directory or file.", required=True)
    parser.add_argument("-p", "--dataset-processor", type=str, default="templeton_aut", help="Name of the dataset to process.")
    parser.add_argument("-m", "--metadata-path", type=str, help="Path to stimulus (item) metadata file.")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory for prepared data.", required=True)
    parser.add_argument("-t", "--task", type=str, default="dt", help="Task name for fMRI data.")
    parser.add_argument("-s", "--space", type=str, default="MNI152NLin2009cAsym", help="Space for fMRI data.")
    parser.add_argument("--tr", type=float, default=2.0, help="Repetition time (TR) in seconds.")
    parser.add_argument("--func-desc", type=str, default="preproc", help="Description for functional images.")
    parser.add_argument("--condition", type=str, default="create", help="Task condition type (e.g., create, recall).")
    parser.add_argument("--roi", type=str, default="whole_brain", help="Region of interest (ROI) for data extraction.")
    parser.add_argument("--roi-resample", action='store_true', help="Whether to resample ROI mask to match functional data.")
    parser.add_argument("--roi-interpolation", type=str, default="nearest", help="Interpolation method for ROI resampling.")
    parser.add_argument("--roi-path", type=str, default=None, help="Path to a custom ROI mask file (NIfTI format). If provided, this will override the `roi` parameter.")
    parser.add_argument("--roi-threshold", type=float, default=0.0, help="Threshold for binarizing continuous ROI masks.")
    parser.add_argument("--exclude-subjects", type=str, nargs='*', default=[], help="List of subject IDs to exclude from processing.")
    parser.add_argument("--sub-stimuli-field", type=str, default="contrast_create_beta_map", help="Field name in subject metadata that contains stimuli file paths.")
    parser.add_argument("--remove-na-responses", action='store_true', help="Whether to remove samples with NaN responses.")
    parser.add_argument("--min-stimuli-rating", type=float, default=None, help="Minimum rating value for stimuli to be included in the dataset.")
    parser.add_argument("--max-stimuli-rating", type=float, default=None, help="Maximum rating value for stimuli to be included in the dataset.")
    parser.add_argument("--min-subject-rating", type=float, default=None, help="Minimum average rating value for subjects to be included in the dataset.")
    parser.add_argument("--max-subject-rating", type=float, default=None, help="Maximum average rating value for subjects to be included in the dataset.")
    parser.add_argument("--skip-dim-mismatch-filter", action='store_true', help="Whether to skip filtering samples with dimension mismatch.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output dataset name.")

    args = parser.parse_args()

    brain_data = None

    tr_value = get_tr_value(args.datapath, space=args.space, task=args.task, func_desc=args.func_desc)
    if tr_value is not None:
        print(f"TR value found in metadata: {tr_value} seconds")
    else:
        print("No TR value found in metadata, using provided tr value:", args.tr)
        tr_value = args.tr

    if args.dataset_processor in DATASET_MAP:
        if args.dataset_processor not in DATASET_MAP:
            raise ValueError(f"Dataset '{args.dataset_processor}' is not supported. Available datasets: {list(DATASET_MAP.keys())}")
        
        brain_data = DATASET_MAP[args.dataset_processor](
            datapath=args.datapath,
            metadata_path=args.metadata_path,
            task=args.task,
            space=args.space,
            func_desc=args.func_desc,
            condition=args.condition,
            tr=tr_value,
            roi=args.roi,
            roi_resample=args.roi_resample,
            roi_interpolation=args.roi_interpolation,
            roi_path=args.roi_path,
            roi_threshold=args.roi_threshold,
            exclude_subjects=args.exclude_subjects,
            sub_stimuli_field=args.sub_stimuli_field
        )

    brain_data = post_filter_brain_data(
        brain_data if brain_data is not None else args.datapath,
        exclude_subjects=args.exclude_subjects,
        remove_na_responses=args.remove_na_responses,
        min_stimuli_rating=args.min_stimuli_rating,
        max_stimuli_rating=args.max_stimuli_rating,
        min_subject_rating=args.min_subject_rating,
        max_subject_rating=args.max_subject_rating,
        skip_dim_mismatch_filter=args.skip_dim_mismatch_filter
    )

    dataset_name = args.datapath.split('/')[-1]
    if dataset_name.endswith('.json'):
        dataset_name = dataset_name.replace('.json', '')

    if args.dataset_processor in ["post_filter", "add_rating"]:
        if args.dataset_processor == "post_filter":
            if "post_filtered" not in dataset_name:
                dataset_name = f"{dataset_name}_post_filtered"
        else:
            if "with_ratings" not in dataset_name:
                dataset_name = f"{dataset_name}_with_ratings"
        output_path = os.path.join(args.output_dir, f"{dataset_name}{args.suffix}.json")
    else:
        output_path = os.path.join(args.output_dir, f"{dataset_name}_{args.task}_{args.condition}{args.suffix}.json")

    output_data = {
        "metadata": {
            "output_path": str(output_path),
            "datapath": args.datapath,
            "dataset_processor": args.dataset_processor,
            "dataset_name": dataset_name,
            "metadata_path": args.metadata_path,
            "output_dir": args.output_dir,
            "task": args.task,
            "space": args.space,
            "func_desc": args.func_desc,
            "condition": args.condition,
            "tr": tr_value,
            "roi": args.roi,
            "roi_resample": args.roi_resample,
            "roi_interpolation": args.roi_interpolation,
            "roi_path": args.roi_path,
            "roi_threshold": args.roi_threshold,
            "num_subjects": len(set([d['subject_id'] for d in brain_data])),
            "num_items": len(set([d['stimuli'] for d in brain_data])),
            "total_samples": len(brain_data),
            "sub_stimuli_field": args.sub_stimuli_field,
            "remove_na_responses": args.remove_na_responses,
            "min_stimuli_rating": args.min_stimuli_rating,
            "max_stimuli_rating": args.max_stimuli_rating,
            "min_subject_rating": args.min_subject_rating,
            "max_subject_rating": args.max_subject_rating,
            "skip_dim_mismatch_filter": args.skip_dim_mismatch_filter
        },
        "data": brain_data
    }

    fmri_dir = os.path.join(args.output_dir, "fmri_data")
    os.makedirs(fmri_dir, exist_ok=True)

    for subject_data in output_data["data"]:
        if "fmri_data" in subject_data:
            subject_datapath = os.path.join(fmri_dir, f"{subject_data['id']}_fmri_data.npy")
            np.save(subject_datapath, subject_data['fmri_data'])
            subject_data.pop('fmri_data')
            subject_data["fmri_path"] = "/".join(subject_datapath.split("/")[-2:])

    write_json(output_data, output_path)
    print(f"Saved data to '{output_path}'")
    print("Human brain fMRI data preparation completed successfully.")


if __name__ == "__main__":
    main()