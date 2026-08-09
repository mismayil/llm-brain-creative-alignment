import argparse
import numpy as np
import pathlib
import re
from scipy.stats import median_abs_deviation

from cadabra.utils import read_json, write_json
from cadabra.alignment.alignment_utils import apply_noise_ceiling, compute_alignment_metrics

def adjust_for_noise_ceiling(alignment_results_path, noise_ceiling_results_path):
    alignment_results = read_json(alignment_results_path)
    noise_ceiling_results = read_json(noise_ceiling_results_path)
    alignment_data_path = None

    if isinstance(alignment_results["data"], str):
        alignment_data_path = alignment_results["data"]
        alignment_results["data"] = {"raw_alignment_data_path": alignment_data_path}
    elif isinstance(alignment_results["data"], dict) and "raw_alignment_data_path" in alignment_results["data"]:
        alignment_data_path = alignment_results["data"]["raw_alignment_data_path"]
    else:
        raise ValueError("Alignment data does not contain a valid data path.")
    
    alignment_data = np.load(pathlib.Path(alignment_results_path).parent / alignment_data_path)
    noise_ceiling_data = np.load(pathlib.Path(noise_ceiling_results_path).parent / noise_ceiling_results["data"]["noise_ceiling_path"])
    
    adjusted_alignment_data = apply_noise_ceiling(alignment_data, noise_ceiling_data)
    num_nan_voxels = np.sum(np.isnan(adjusted_alignment_data), axis=-1)
    print(f"Number of voxels set to NaN after noise ceiling adjustment: {num_nan_voxels}")

    adjusted_alignment_metrics = compute_alignment_metrics(adjusted_alignment_data)

    noise_ceiling_run_id = re.search(r'(\d{8}_\d{6})', pathlib.Path(noise_ceiling_results_path).stem).group(1)

    alignment_results["metadata"][f"noise_ceiling_path_{noise_ceiling_run_id}"] = noise_ceiling_results_path
    alignment_results["metrics"][f"noise_ceiling_adjusted_{noise_ceiling_run_id}"] = adjusted_alignment_metrics
    
    adjusted_data_path = f"{pathlib.Path(alignment_data_path).stem}_ns_adjusted_{noise_ceiling_run_id}.npy"
    adjusted_data_full_path = pathlib.Path(alignment_results_path).parent / adjusted_data_path
    np.save(adjusted_data_full_path, adjusted_alignment_data)
    alignment_results["data"][f"noise_ceiling_adjusted_data_path_{noise_ceiling_run_id}"] = adjusted_data_path

    write_json(alignment_results, alignment_results_path)
    print(f"Updated alignment results saved to {alignment_results_path}")

def main():
    parser = argparse.ArgumentParser(description="Update alignment results (e.g. adjust for noise ceiling).")
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Path to the input alignment results file")
    parser.add_argument("-n", "--noise-ceiling-path", type=str, help="Path to the noise ceiling data file")

    args = parser.parse_args()

    if args.noise_ceiling_path:
        adjust_for_noise_ceiling(args.input_path, args.noise_ceiling_path)
    
    print("Update complete.")


if __name__ == "__main__":
    main()