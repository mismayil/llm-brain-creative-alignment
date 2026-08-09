import argparse
from nilearn import image as nimg
import os
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel

from cadabra.utils import keep_most_frequent_size, write_json, generate_datetime_id, read_json

def run_second_level_glm(flm_results_path, output_dir,
                         subject_field="subject", effects_map_field="contrast_z_map", 
                         smoothing_fwhm=6.0):
    """
    Run second-level GLM analysis on first-level results.
    """
    flm_results = read_json(flm_results_path)

    print(f"Loaded {len(flm_results['data'])} first-level results from {flm_results_path}")
    print(f"Using subject field: {subject_field}")
    print(f"Using effects map field: {effects_map_field}")

    print("Filtering images to keep only those with the most frequent size...")
    flm_images = [nimg.load_img(result[effects_map_field]).get_fdata() for result in flm_results["data"]]
    flm_images, frequent_indices = keep_most_frequent_size(flm_images, return_indices=True)
    flm_results_data = [result for i, result in enumerate(flm_results["data"]) if i in frequent_indices]

    first_level_contrast = effects_map_field.replace("_z_map", "").replace("_beta_map", "")
    print(f"Using first-level contrast: {first_level_contrast}")
    print(f"Number of subjects after filtering: {len(flm_results_data)}")
    if len(flm_results_data) == 0:
        raise ValueError("No valid first-level results found after filtering.")

    second_level_input = pd.DataFrame({
        "subject_label": [result[subject_field] for result in flm_results_data],
        "map_name": [first_level_contrast] * len(flm_results_data),
        "effects_map_path": [
            result[effects_map_field] for result in flm_results_data
        ]
    })

    second_level_model = SecondLevelModel(
        smoothing_fwhm=smoothing_fwhm,        # optional smoothing
    )

    print("Fitting second-level GLM model...")
    second_level_model = second_level_model.fit(
        second_level_input
    )

    result = {}

    for output_type, output_name in [("z_score", "z_map"), ("effect_size", "beta_map")]:
        print(f"Computing contrast: {output_name}...")
        output_img = second_level_model.compute_contrast(
            second_level_contrast="intercept",  # the mean effect
            first_level_contrast=first_level_contrast,
            output_type=output_type
        )
        output_path = os.path.join(output_dir, f"second_level_{first_level_contrast}_{output_name}.nii.gz")
        output_img.to_filename(output_path)
        print(f"Saved {output_name} to {output_path}")
        result[output_name] = output_path

    return result


def main():
    parser = argparse.ArgumentParser(description="Run second-level GLM analysis on first-level data.")

    parser.add_argument("-i", "--flm-results-path", type=str, help="Path to First-Level GLM results.", required=True)
    parser.add_argument("-o", "--output-dir", type=str, help="Path to output directory for GLM results.", required=True)
    parser.add_argument("--subject-field", type=str, default="subject", help="Field in the JSON to use as subject identifier.")
    parser.add_argument("--effects-map-field", type=str, default="create_vs_object_z_map", help="Field in the JSON to use as effects map path.")
    parser.add_argument("--smoothing-fwhm", type=float, default=6.0, help="FWHM for smoothing the second-level images.")

    args = parser.parse_args()

    run_id = generate_datetime_id()
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    results = run_second_level_glm(args.flm_results_path, output_dir,
                                    subject_field=args.subject_field,
                                    effects_map_field=args.effects_map_field,
                                    smoothing_fwhm=args.smoothing_fwhm)

    output_path = os.path.join(output_dir, f"second_level_glm_results_{run_id}.json")

    output_data = {
        "metadata": {
            "output_path": output_path,
            "run_id": run_id,
            "flm_results_path": args.flm_results_path,
            "subject_field": args.subject_field,
            "effects_map_field": args.effects_map_field,
            "output_dir": output_dir,
            "smoothing_fwhm": args.smoothing_fwhm
        },
        "data": results 
    }

    write_json(output_data, )

    print("Second-level GLM analysis completed successfully.")

if __name__ == "__main__":
    main()