from dotenv import load_dotenv
import hashlib
import numpy as np
from pathlib import Path

load_dotenv()

from cadabra.utils import read_json, write_json, get_dict_value
from cadabra.alignment.alignment_utils import get_alignment_runs


rating_threshold = 2.25

nc_thresholds=(0, 0.1, 0.2)
brains=(
    f"yeo_dmn.*post_filtered_rating_ge_{rating_threshold}.json", 
    f"yeo_dmn.*post_filtered_rating_lt_{rating_threshold}.json"
)
model_networks=(
    "layers",
)
activation_modes=(
    "prompt",
    "model_resp",
    "short_prompt",
    "short_model_resp"
)
model_data_samplings=(
    "time:mean::layer:1:",
    "time:-1::layer:1:"
)
network_poolings=(
    ".*pmean.*",
    # ".*plast-token.*"
)
models=(
    "llama-3.1-8b-instruct",
    "llama-3.1-8b",
    "llama-3.1-minitaur-8b",
    "crpo-llama-3.1-8b-instruct-cre",
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-qwen-7b",
    "qwen2.5-math-7b",
    "qwen2.5-7b",
    "qwen2.5-7b-instruct",
    "crpo-sft-llama-3.1-8b-instruct",
    "crpo-dpo-llama-3.1-8b-instruct"
)
datasets=(
    f".*templeton_aut_create_eval_data_({('|').join(models)})_.*",
    f".*templeton_aut_create_eval_data_({('|').join(models)})_.*",
    f".*templeton_aut_create_short_eval_data_({('|').join(models)})_.*",
    f".*templeton_aut_create_short_eval_data_({('|').join(models)})_.*"
)

high_low_cre_results = []

for nc in nc_thresholds:
    for model_network in model_networks:
        for activation_mode, dataset in zip(activation_modes, datasets):
            for model_data_sampling in model_data_samplings:
                for network_pooling in network_poolings:
                    high_cre_brain, low_cre_brain = brains
                    high_cre_runs = get_alignment_runs(
                        alignment_method="rsa_per_subject",
                        nc_threshold=nc,
                        brain_network=high_cre_brain,
                        dataset=dataset,
                        model_data_sampling=model_data_sampling,
                        regression_target=None,
                        model_network=model_network,
                        network_pooling=network_pooling,
                        wandb_project="cadabra-alignments",
                        activation_mode=activation_mode
                    )
                    low_cre_runs = get_alignment_runs(
                        alignment_method="rsa_per_subject",
                        nc_threshold=nc,
                        brain_network=low_cre_brain,
                        dataset=dataset,
                        model_data_sampling=model_data_sampling,
                        regression_target=None,
                        model_network=model_network,
                        network_pooling=network_pooling,
                        wandb_project="cadabra-alignments",
                        activation_mode=activation_mode
                    )

                    print(f"Found {len(high_cre_runs)} high CRE runs and {len(low_cre_runs)} low CRE runs for nc_threshold={nc}, model_network={model_network}, activation_mode={activation_mode}, model_data_sampling={model_data_sampling}, network_pooling={network_pooling}")

                    for high_run in high_cre_runs:
                        model_name = high_run.config["model_name"]
                        for low_run in low_cre_runs:
                            if low_run.config["model_name"] == model_name:
                                high_run_best_layer = get_dict_value(high_run.summary_metrics, "best_layer.layer_num")
                                high_run_output_path = Path(get_dict_value(high_run.config, "output_path")).resolve()
                                high_run_outputs = read_json(high_run_output_path)
                                high_run_subjects = high_run_outputs["metadata"]["subjects"]
                                high_run_subjects_alignments = np.load(high_run_output_path.parent / high_run_outputs["data"]["noise_ceiling_adjusted_scores_path"])
                                high_run_best_layer_alignments = high_run_subjects_alignments[0, high_run_best_layer, :]
                                high_cre_pred = get_dict_value(high_run.summary_metrics, "noise_ceiling_adjusted.best_layer.median_pred")

                                low_run_best_layer = get_dict_value(low_run.summary_metrics, "best_layer.layer_num")
                                low_run_output_path = Path(get_dict_value(low_run.config, "output_path")).resolve()
                                low_run_outputs = read_json(low_run_output_path)
                                low_run_subjects = low_run_outputs["metadata"]["subjects"]
                                low_run_subjects_alignments = np.load(low_run_output_path.parent / low_run_outputs["data"]["noise_ceiling_adjusted_scores_path"])
                                low_run_best_layer_alignments = low_run_subjects_alignments[0, low_run_best_layer, :]
                                low_cre_pred = get_dict_value(low_run.summary_metrics, "noise_ceiling_adjusted.best_layer.median_pred")
                                
                                high_low_cre_results.append({
                                    "model_name": model_name,
                                    "nc_threshold": nc,
                                    "model_network": model_network,
                                    "network_pooling": network_pooling,
                                    "model_data_sampling": model_data_sampling,
                                    "activation_mode": activation_mode,
                                    "high_cre_pred": high_cre_pred,
                                    "low_cre_pred": low_cre_pred,
                                    "high_sub_low_pred_diff": high_cre_pred - low_cre_pred,
                                    "high_run_subjects": high_run_subjects,
                                    "low_run_subjects": low_run_subjects,
                                    "high_run_best_layer_alignments": high_run_best_layer_alignments.tolist(),
                                    "low_run_best_layer_alignments": low_run_best_layer_alignments.tolist()
                                })

print(f"Total high-low CRE pairs found: {len(high_low_cre_results)}")
write_json(high_low_cre_results, f"experiments/templeton_aut/data/high_low_cre_results_{rating_threshold}.json")