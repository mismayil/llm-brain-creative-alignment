from dotenv import load_dotenv
import hashlib
import numpy as np
from pathlib import Path

load_dotenv()

from cadabra.utils import read_json, write_json, get_dict_value, extract_model_size
from cadabra.alignment.alignment_utils import get_alignment_runs

GEMINI_AUT_SCORING_PATHS = {
    "llama-3.1-8b-instruct": "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260306_140725/aut_scoring/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260306_140725_aut_scoring_gemini-3-flash-preview_20260322_191202.json",
    "crpo-llama-3.1-8b-instruct-cre": "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260306_140819/aut_scoring/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260306_140819_aut_scoring_gemini-3-flash-preview_20260322_191307.json",
    "gemma-3-270m-it": "../outputs/templeton_aut/aut/gemma-3-270m-it/20260316_153058/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260316_153058_aut_scoring_gemini-3-flash-preview_20260322_191429.json",
    "gemma-3-1b-it": "../outputs/templeton_aut/aut/gemma-3-1b-it/20260316_153639/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260316_153639_aut_scoring_gemini-3-flash-preview_20260322_191634.json",
    "gemma-3-4b-it": "../outputs/templeton_aut/aut/gemma-3-4b-it/20260316_154445/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260316_154445_aut_scoring_gemini-3-flash-preview_20260322_191843.json",
    "gemma-3-12b-it": "../outputs/templeton_aut/aut/gemma-3-12b-it/20260316_155546/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260316_155546_aut_scoring_gemini-3-flash-preview_20260322_192034.json",
    "gemma-3-27b-it": "../outputs/templeton_aut/aut/gemma-3-27b-it/20260316_161156/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260316_161156_aut_scoring_gemini-3-flash-preview_20260322_192142.json",
    "llama-3.2-1b-instruct": "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260316_163439/aut_scoring/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260316_163439_aut_scoring_gemini-3-flash-preview_20260322_192245.json",
    "llama-3.2-3b-instruct": "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260316_163502/aut_scoring/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260316_163502_aut_scoring_gemini-3-flash-preview_20260322_192432.json",
    "olmo-3.1-32b-instruct": "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260322_145141/aut_scoring/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260322_145141_aut_scoring_gemini-3-flash-preview_20260322_192601.json",
    "llama-3.1-70b-instruct": "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260322_145227/aut_scoring/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260322_145227_aut_scoring_gemini-3-flash-preview_20260322_192725.json",
    "llama-3.1-8b": "../outputs/templeton_aut/aut/llama-3.1-8b/20260327_200542/templeton_aut_create_short_eval_data_llama-3.1-8b_20260327_200542.json",
    "llama-3.1-minitaur-8b": "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260327_201250/templeton_aut_create_short_eval_data_llama-3.1-minitaur-8b_20260327_201250.json",
    "qwen2.5-32b-instruct": "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260328_210352/templeton_aut_create_short_eval_data_qwen2.5-32b-instruct_20260328_210352.json",
    "qwen2.5-72b-instruct": "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260328_210430/templeton_aut_create_short_eval_data_qwen2.5-72b-instruct_20260328_210430.json",
    "deepseek-r1-distill-llama-8b": "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260328_210527/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-8b_20260328_210527.json",
    "deepseek-r1-distill-llama-70b": "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260328_211319/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-70b_20260328_211319.json",
    "falcon-40b-instruct": "../outputs/templeton_aut/aut/falcon-40b-instruct/20260328_214322/templeton_aut_create_short_eval_data_falcon-40b-instruct_20260328_214322.json",
    "qwen2.5-14b-instruct": "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260328_214636/templeton_aut_create_short_eval_data_qwen2.5-14b-instruct_20260328_214636.json",
    "qwen2.5-7b-instruct": "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260527_111143/templeton_aut_create_short_eval_data_qwen2.5-7b-instruct_20260527_111143.json",
    "mistral-7b-instruct-v0.3": "../outputs/templeton_aut/aut/mistral-7b-instruct-v0.3/20260325_180953/templeton_aut_create_short_eval_data_mistral-7b-instruct-v0.3_20260325_180953.json",
    "olmo-3-7b-instruct": "../outputs/templeton_aut/aut/olmo-3-7b-instruct/20260801_100207/templeton_aut_create_short_eval_data_olmo-3-7b-instruct_20260801_100207.json",
    "qwen2.5-0.5b-instruct": "../outputs/templeton_aut/aut/qwen2.5-0.5b-instruct/20260801_161344/templeton_aut_create_short_eval_data_qwen2.5-0.5b-instruct_20260801_161344.json",
    "qwen2.5-1.5b-instruct": "../outputs/templeton_aut/aut/qwen2.5-1.5b-instruct/20260801_161435/templeton_aut_create_short_eval_data_qwen2.5-1.5b-instruct_20260801_161435.json",
    "qwen2.5-3b-instruct": "../outputs/templeton_aut/aut/qwen2.5-3b-instruct/20260801_161453/templeton_aut_create_short_eval_data_qwen2.5-3b-instruct_20260801_161453.json"
}

nc_thresholds=[0]
brains=(
    "yeo_dmn.*dt_create_with_ratings.json", 
    "yeo_fp.*dt_create_with_ratings.json",
    "yeo_som.*dt_create.json",
    "yeo_dmn.*dt_object.json", 
    "yeo_fp.*dt_object.json",
)
model_networks=(
    "layers",
)
activation_modes=(
    "prompt",
    "model_resp"
)
model_data_samplings=(
    "time:mean::layer:1:",
    "time:-1::layer:1:"
)

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "CNCL-Penn-State/CrPO-llama-3.1-8b-instruct-cre",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "allenai/Olmo-3.1-32B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "tiiuae/falcon-40b-instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
]
models = [model.split("/")[-1].lower() for model in models]
datasets=(
    f".*templeton_aut_create_eval_data_({('|').join(models)})_.*",
    f".*templeton_aut_object_eval_data_({('|').join(models)})_.*",
    f".*templeton_aut_create_short_eval_data_({('|').join(models)})_.*",
)

main_results = []

for brain in brains:
    for nc in nc_thresholds:
        for model_network in model_networks:
            for activation_mode in activation_modes:
                for dataset in datasets:
                    for model_data_sampling in model_data_samplings:
                        runs = get_alignment_runs(
                            brain_network=brain,
                            nc_threshold=nc,
                            model_network=model_network,
                            activation_mode=activation_mode,
                            model_data_sampling=model_data_sampling,
                            dataset=dataset,
                            force_refresh=True
                        )

                        print(f"Found {len(runs)} runs for nc_threshold={nc}, model_network={model_network}, activation_mode={activation_mode}, model_data_sampling={model_data_sampling}")

                        for run in runs:
                            model_name = run.config["model_name"]
                            model_size = extract_model_size(model_name) / 1e9
                            best_layer = get_dict_value(run.summary_metrics, "best_layer.layer_num")
                            num_total_layers = get_dict_value(run.summary_metrics, "last_layer.layer_num")
                            relative_depth = best_layer / num_total_layers if num_total_layers != 0 else 0
                            alignment = get_dict_value(run.summary_metrics, "noise_ceiling_adjusted.best_layer.median_pred")
                            
                            aut_scoring_path = GEMINI_AUT_SCORING_PATHS.get(model_name)
                            if aut_scoring_path is None:
                                print(f"No Gemini AUT scoring path found for model '{model_name}'.")
                                continue
                            else:
                                aut_scoring_results = read_json(aut_scoring_path)
                                if "aut_scoring" in aut_scoring_path:
                                    aut_score = np.nanmean([int(sample["output"].strip()[0]) for sample in aut_scoring_results["data"]])
                                else:
                                    aut_score = np.nanmean([sample["gemini_aut_score"] for sample in aut_scoring_results["data"]])
                            
                            main_results.append({
                                "model_name": model_name,
                                "nc_threshold": nc,
                                "brain_network": brain,
                                "model_network": model_network,
                                "model_data_sampling": model_data_sampling,
                                "activation_mode": activation_mode,
                                "alignment": alignment,
                                "best_layer": best_layer,
                                "total_layers": num_total_layers,
                                "relative_depth": relative_depth,
                                "model_size_b": model_size,
                                "aut_score": aut_score,
                                "dataset": dataset
                            })

print(f"Total main results found: {len(main_results)}")
write_json(main_results, f"experiments/templeton_aut/data/main_results.json")