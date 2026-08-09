import pathlib
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import hashlib
import numpy as np

load_dotenv()  # Load environment variables from .env file

from cadabra.utils import extract_model_size, generate_datetime_id, write_json, read_json, get_dict_value
from cadabra.alignment.alignment_utils import get_alignment_runs

LEGEND_MAPPING = {
    "model_name": "Model Name",
    "noise_ceiling_adjusted.best_layer.median_pred": "Predictivity"
}

def visualize_alignment_by_model_size(runs, metric="median_pred", output_path=None):
    viz_data = []

    for run in runs:
        model_name = run.config["model_name"]
        model_size = extract_model_size(model_name)
        metric_value = get_dict_value(run.summary_metrics, metric)

        if model_size is None or metric_value is None:
            continue

        viz_data.append(
            {
                "model_name": model_name,
                "model_size_b": model_size / 1e9,
                "metric_value": metric_value,
            }
        )

    if not viz_data:
        print(f"No valid data found for metric '{metric}'. Skipping visualization.")
        return

    df = pd.DataFrame(viz_data)
    df = df.sort_values(["model_size_b", "model_name"])

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("gist_earth", n_colors=max(df["model_name"].nunique(), 3))

    fig, ax = plt.subplots(figsize=(9.6, 6))
    
    # Use scatterplot with style for different markers per model
    sns.scatterplot(
        data=df,
        x="model_size_b",
        y="metric_value",
        hue="model_name",
        style="model_name",
        palette=palette,
        alpha=0.9,
        s=80,
        ax=ax,
    )
    
    # Add one global regression line
    sns.regplot(
        data=df,
        x="model_size_b",
        y="metric_value",
        scatter=False,
        ax=ax,
        color="#8c6d46",
        line_kws={"linewidth": 2.0, "alpha": 0.95},
    )

    ax.set_facecolor("#f5f0e6")
    ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)

    ax.set_title(LEGEND_MAPPING.get(metric, metric), fontsize=14, pad=12)
    ax.set_xlabel("Model Size")
    ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}B"))
    
    # Position legend on the right
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="#f5f0e6")

    plt.close(fig)

AUT_SCORING_PATHS = {
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
    "llama-3.1-70b-instruct": "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260322_145227/aut_scoring/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260322_145227_aut_scoring_gemini-3-flash-preview_20260322_192725.json"
}

def visualize_alignment_by_aut_score(runs, metric="aut_score", output_path=None):
    viz_data = []

    for run in runs:
        model_name = run.config["model_name"]
        aut_scoring_path = AUT_SCORING_PATHS.get(model_name)
        if aut_scoring_path is None:
            print(f"No AUT scoring path found for model '{model_name}'. Skipping.")
            continue
        aut_scoring_results = read_json(aut_scoring_path)
        aut_score = np.mean([int(sample["output"].strip()[0]) for sample in aut_scoring_results["data"]])
        metric_value = get_dict_value(run.summary_metrics, metric)

        viz_data.append(
            {
                "model_name": model_name,
                "aut_score": aut_score,
                "metric_value": metric_value,
            }
        )

    if not viz_data:
        print(f"No valid data found for metric '{metric}'. Skipping visualization.")
        return

    df = pd.DataFrame(viz_data)
    df = df.sort_values(["aut_score", "model_name"])

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("gist_earth", n_colors=max(df["model_name"].nunique(), 3))

    fig, ax = plt.subplots(figsize=(9.6, 6))
    
    # Use scatterplot with style for different markers per model
    sns.scatterplot(
        data=df,
        x="aut_score",
        y="metric_value",
        hue="model_name",
        style="model_name",
        palette=palette,
        alpha=0.9,
        s=80,
        ax=ax,
    )
    
    # Add one global regression line
    sns.regplot(
        data=df,
        x="aut_score",
        y="metric_value",
        scatter=False,
        ax=ax,
        color="#8c6d46",
        line_kws={"linewidth": 2.0, "alpha": 0.95},
    )

    ax.set_facecolor("#f5f0e6")
    ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)

    ax.set_title(LEGEND_MAPPING.get(metric, metric), fontsize=14, pad=12)
    ax.set_xlabel("AUT Score")
    ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
    
    # Position legend on the right
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="#f5f0e6")

    plt.close(fig)

VIZ_MAPPING = {
    "alignment_by_model_size": visualize_alignment_by_model_size,
    "alignment_by_aut_score": visualize_alignment_by_aut_score,
}

@hydra.main(version_base=None, config_path="configs", config_name="brain_alignment_viz")
def main(config: DictConfig):
    latest_runs = get_alignment_runs(
        alignment_method=config.alignment_method,
        nc_threshold=config.nc_threshold,
        brain_network=config.brain_network,
        dataset=config.dataset,
        model_data_sampling=config.model_data_sampling,
        regression_target=config.regression_target,
        model_network=config.model_network,
        network_pooling=config.network_pooling,
        wandb_project=config.wandb_project,
        activation_mode=config.activation_mode,
    )

    print(f"After keeping only the latest run per config, {len(latest_runs)} runs remain.")

    if latest_runs:
        viz_func = VIZ_MAPPING.get(config.viz_name)
        if viz_func is not None:
            viz_run_id = f"viz_{config.viz_name}_{generate_datetime_id()}"
            output_dir = pathlib.Path(config.output_dir) / viz_run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json({"config": OmegaConf.to_container(config)}, output_dir / "config.json")
            
            for metric in config.metrics:
                viz_func(latest_runs, metric=metric, output_path=output_dir / f"{metric}.{config.output_format}")
            
            print(f"Visualization completed. Results saved to {output_dir}")
        else:
            print(f"Visualization '{config.viz_name}' not recognized.")

if __name__ == "__main__":
    main()