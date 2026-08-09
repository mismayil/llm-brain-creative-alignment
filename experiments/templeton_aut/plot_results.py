from dotenv import load_dotenv
import hashlib
import json
import numpy as np
import pathlib
from types import SimpleNamespace
from itertools import product

load_dotenv()

from cadabra.utils import get_wandb_runs, read_json
from cadabra.alignment.alignment_utils import get_alignment_runs

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from cadabra.utils import get_wandb_runs, extract_model_size, generate_datetime_id, write_json, read_json, get_dict_value
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

# Figure canvas stays white; the beige tint is applied to the axes area only.
plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 16,
})

metric = "noise_ceiling_adjusted.best_layer.median_pred"
LEGEND_MAPPING = {
    "model_name": "Model Name",
    "noise_ceiling_adjusted.best_layer.median_pred": "RSA Predictivity"
}

NETWORK_MAPPING = {
    "layers": "Model Layers",
    "network": "Model Creativity Network",
}

ACTIVATION_MODE_MAPPING = {
    "prompt": "Prompt Activations",
    "model_resp": "Prompt+Response Activations",
}

MODEL_DATA_SAMPLING_MAPPING = {
    "time:-1::layer:1:": "last token",
    "time:mean::layer:1:": "mean token",
}

BRAIN_NETWORK_MAPPING = {
    "yeo_dmn.*dt_create_with_ratings.json": "DMN",
    "yeo_fp.*dt_create_with_ratings.json": "FPN",
    "yeo_som.*dt_create.json": "SOM",
    "yeo_dmn.*dt_object.json": "DMN",
    "yeo_fp.*dt_object.json": "FPN",
    "yeo_som.*dt_object.json": "SOM",
}

TASK_MAPPING = {
    "yeo_dmn.*dt_create_with_ratings.json": "AUT",
    "yeo_fp.*dt_create_with_ratings.json": "AUT",
    "yeo_som.*dt_create.json": "AUT",
    "yeo_dmn.*dt_object.json": "OCT",
    "yeo_fp.*dt_object.json": "OCT",
    "yeo_som.*dt_object.json": "OCT",
}

def plot_model_size_results(filters_lst, runs_lst, metric, models=None, output_path=None):
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    all_model_names = set()
    all_df_list = []

    # Process each filter and collect data
    for idx, (filter_dict, runs) in enumerate(zip(filters_lst, runs_lst)):
        ax = axes[idx]

        # Extract data
        viz_data = []
        for run in runs:
            model_name = run.config["model_name"]
            if models and model_name not in models:
                print(f"Skipping model '{model_name}'.")
                continue
            model_size = extract_model_size(model_name)
            metric_value = get_dict_value(run.summary_metrics, metric)

            if model_size is None or metric_value is None:
                continue

            viz_data.append({
                "model_name": model_name,
                "model_size_b": model_size / 1e9,
                "metric_value": metric_value,
            })
            all_model_names.add(model_name)

        if not viz_data:
            print(f"No valid data found for filter {idx}. Skipping.")
            continue

        df = pd.DataFrame(viz_data)
        df = df.sort_values(["model_size_b", "model_name"])
        all_df_list.append((ax, df, filter_dict))

    # Determine color and marker mappings based on all unique model names
    sorted_models = sorted(all_model_names)
    palette = sns.color_palette("gist_earth", n_colors=max(len(sorted_models), 3))
    model_name_to_color = {name: palette[i] for i, name in enumerate(sorted_models)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    model_name_to_marker = {name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(sorted_models)}

    # Plot each subplot
    for ax, df, filter_dict in all_df_list:
        # Plot scatter points with model-specific markers
        for model_name in df["model_name"].unique():
            data_subset = df[df["model_name"] == model_name]
            ax.scatter(
                data_subset["model_size_b"],
                data_subset["metric_value"],
                label=model_name,
                color=model_name_to_color[model_name],
                marker=model_name_to_marker[model_name],
                alpha=0.9,
                s=80,
                edgecolors="black",
                linewidths=0.35,
            )

        # Add regression line
        sns.regplot(
            data=df,
            x="model_size_b",
            y="metric_value",
            scatter=False,
            ax=ax,
            color="#8c6d46",
            line_kws={"linewidth": 2.0, "alpha": 0.95},
        )

        # Styling
        ax.set_facecolor("#f5f0e6")
        ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)

        # Title with model network and activation mode and the correlation coefficient and p value
        model_network = NETWORK_MAPPING.get(filter_dict["model_network"], filter_dict["model_network"])
        activation_mode = ACTIVATION_MODE_MAPPING.get(filter_dict["activation_mode"], filter_dict["activation_mode"])
        sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(filter_dict["model_data_sampling"], filter_dict["model_data_sampling"])
        corr_coef, p_value = pearsonr(df["model_size_b"], df["metric_value"])
        ax.set_title(f"{activation_mode} (pooling: {sampling_label})\n(r={corr_coef:.2f}, p={p_value:.2f})", fontsize=15, pad=12)

        ax.set_xlabel("Model Size (B)")
        ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}B"))

    brain_network = filters_lst[0]["brain_network"] if filters_lst else None
    fig.suptitle(f"Brain Alignment\n{BRAIN_NETWORK_MAPPING.get(brain_network, brain_network)} | {TASK_MAPPING.get(brain_network, brain_network)} | Model Size", fontsize=17, y=0.98)

    # Create a shared, wrapped legend at the bottom
    legend_handles = [
        Line2D(
            [0], [0],
            marker=model_name_to_marker[model_name],
            linestyle="",
            markerfacecolor=model_name_to_color[model_name],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=10,
            color="none",
        )
        for model_name in sorted_models
    ]
    legend_labels = sorted_models
    legend_ncol = min(4, max(1, len(legend_labels)))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, 0.1),
        frameon=True,
        title="Model",
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")

    plt.show()

def plot_best_layer_results(filter_dict, runs, metric, models=None, output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_dist, ax_scatter = axes

    viz_data = []
    for run in runs:
        model_name = run.config["model_name"]
        if models and model_name not in models:
            print(f"Skipping model '{model_name}'.")
            continue

        metric_value = get_dict_value(run.summary_metrics, metric)
        best_layer = get_dict_value(run.summary_metrics, "best_layer.layer_num")
        num_total_layers = get_dict_value(run.summary_metrics, "last_layer.layer_num")

        if best_layer is None or num_total_layers is None or metric_value is None:
            continue

        viz_data.append(
            {
                "model_name": model_name,
                "relative_depth": best_layer / num_total_layers if num_total_layers != 0 else 0,
                "metric_value": metric_value,
            }
        )

    if not viz_data:
        print("No valid data found for the selected runs/models. Skipping plot.")
        return

    df = pd.DataFrame(viz_data)
    df = df.sort_values(["relative_depth", "model_name"])

    sorted_models = sorted(df["model_name"].unique())
    palette = sns.color_palette("gist_earth", n_colors=max(len(sorted_models), 3))
    model_name_to_color = {name: palette[i] for i, name in enumerate(sorted_models)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    model_name_to_marker = {name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(sorted_models)}
    sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(filter_dict["model_data_sampling"], filter_dict["model_data_sampling"])

    # Left: density histogram of relative depth across all included models.
    sns.histplot(
        data=df,
        x="relative_depth",
        stat="density",
        color="#8c6d46",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
        ax=ax_dist,
    )
    ax_dist.set_facecolor("#f5f0e6")
    ax_dist.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)
    ax_dist.set_xlabel("Best Layer Relative Depth")
    ax_dist.set_ylabel("Density")
    ax_dist.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax_dist.set_title(f"Best Layer Relative Depth Distribution (pooling: {sampling_label})", fontsize=15, pad=12)

    # Right: relative depth vs alignment (existing scatter + regression).
    for model_name in df["model_name"].unique():
        data_subset = df[df["model_name"] == model_name]
        ax_scatter.scatter(
            data_subset["relative_depth"],
            data_subset["metric_value"],
            label=model_name,
            color=model_name_to_color[model_name],
            marker=model_name_to_marker[model_name],
            alpha=0.9,
            s=80,
            edgecolors="black",
            linewidths=0.35,
        )

    if len(df) >= 2:
        sns.regplot(
            data=df,
            x="relative_depth",
            y="metric_value",
            scatter=False,
            ax=ax_scatter,
            color="#8c6d46",
            line_kws={"linewidth": 2.0, "alpha": 0.95},
        )

    ax_scatter.set_facecolor("#f5f0e6")
    ax_scatter.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)
    ax_scatter.set_xlabel("Best Layer Relative Depth")
    ax_scatter.set_ylabel(LEGEND_MAPPING.get(metric, metric))
    ax_scatter.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    activation_mode = ACTIVATION_MODE_MAPPING.get(filter_dict["activation_mode"], filter_dict["activation_mode"])
    if len(df) >= 2:
        corr_coef, p_value = pearsonr(df["relative_depth"], df["metric_value"])
        ax_scatter.set_title(f"{activation_mode} (pooling: {sampling_label})\n(r={corr_coef:.2f}, p={p_value:.2f})", fontsize=15, pad=12)
    else:
        ax_scatter.set_title(f"{activation_mode} (pooling: {sampling_label})\n(insufficient data for correlation)", fontsize=15, pad=12)

    brain_network = filter_dict["brain_network"]
    fig.suptitle(f"Brain Alignment\n{BRAIN_NETWORK_MAPPING.get(brain_network, brain_network)} | {TASK_MAPPING.get(brain_network, brain_network)} | Best Layer", fontsize=17, y=0.98)

    legend_handles = [
        Line2D(
            [0], [0],
            marker=model_name_to_marker[model_name],
            linestyle="",
            markerfacecolor=model_name_to_color[model_name],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=10,
            color="none",
        )
        for model_name in sorted_models
    ]
    legend_ncol = min(4, max(1, len(sorted_models)))
    fig.legend(
        legend_handles,
        sorted_models,
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, 0.1),
        frameon=True,
        title="Model",
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")

    plt.show()

def plot_best_layer_dist_comparison(filter_dict1, runs1, filter_dict2, runs2, metric, models=None, output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, filter_dict, runs in zip(axes, [filter_dict1, filter_dict2], [runs1, runs2]):
        viz_data = []
        for run in runs:
            model_name = run.config["model_name"]
            if models and model_name not in models:
                print(f"Skipping model '{model_name}'.")
                continue

            metric_value = get_dict_value(run.summary_metrics, metric)
            best_layer = get_dict_value(run.summary_metrics, "best_layer.layer_num")
            num_total_layers = get_dict_value(run.summary_metrics, "last_layer.layer_num")

            if best_layer is None or num_total_layers is None or metric_value is None:
                continue

            viz_data.append(
                {
                    "model_name": model_name,
                    "relative_depth": best_layer / num_total_layers if num_total_layers != 0 else 0,
                }
            )

        if not viz_data:
            print("No valid data found for the selected runs/models. Skipping subplot.")
            continue

        df = pd.DataFrame(viz_data)
        df = df.sort_values(["relative_depth", "model_name"])

        sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(filter_dict["model_data_sampling"], filter_dict["model_data_sampling"])
        activation_mode = ACTIVATION_MODE_MAPPING.get(filter_dict["activation_mode"], filter_dict["activation_mode"])

        # Density histogram of relative depth across all included models.
        sns.histplot(
            data=df,
            x="relative_depth",
            stat="density",
            color="#8c6d46",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75,
            ax=ax,
        )
        ax.set_facecolor("#f5f0e6")
        ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)
        ax.set_xlabel("Best Layer Relative Depth")
        ax.set_ylabel("Density")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_title(f"{activation_mode} (pooling: {sampling_label})", fontsize=15, pad=12)

    brain_network = filter_dict1["brain_network"]
    fig.suptitle(f"{BRAIN_NETWORK_MAPPING.get(brain_network, brain_network)} | {TASK_MAPPING.get(brain_network, brain_network)} | Best Layer Relative Depth Distribution", fontsize=17, y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")

    plt.show()

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

def plot_aut_score_results(filters_lst, runs_lst, metric, models=None, output_path=None):
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    all_model_names = set()
    all_df_list = []

    # Process each filter and collect data
    for idx, (filter_dict, runs) in enumerate(zip(filters_lst, runs_lst)):
        ax = axes[idx]

        # Extract data
        viz_data = []
        for run in runs:
            model_name = run.config["model_name"]
            if models and model_name not in models:
                continue

            aut_scoring_path = GEMINI_AUT_SCORING_PATHS.get(model_name)
            if aut_scoring_path is None:
                print(f"No AUT scoring path found for model '{model_name}'. Skipping.")
                continue
            aut_scoring_results = read_json(aut_scoring_path)
            if "aut_scoring" in aut_scoring_path: 
                aut_score = np.nanmean([int(sample["output"].strip()[0]) for sample in aut_scoring_results["data"]])
            else:
                aut_score = np.nanmean([sample["gemini_aut_score"] for sample in aut_scoring_results["data"]])
            model_size = extract_model_size(model_name)
            metric_value = get_dict_value(run.summary_metrics, metric)

            if model_size is None or metric_value is None:
                continue

            viz_data.append({
                "model_name": model_name,
                "aut_score": aut_score,
                "metric_value": metric_value,
            })
            all_model_names.add(model_name)

        if not viz_data:
            print(f"No valid data found for filter {idx}. Skipping.")
            continue

        df = pd.DataFrame(viz_data)
        df = df.sort_values(["aut_score", "model_name"])
        all_df_list.append((ax, df, filter_dict))

    # Determine color and marker mappings based on all unique model names
    sorted_models = sorted(all_model_names)
    palette = sns.color_palette("gist_earth", n_colors=max(len(sorted_models), 3))
    model_name_to_color = {name: palette[i] for i, name in enumerate(sorted_models)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    model_name_to_marker = {name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(sorted_models)}

    # Plot each subplot
    for ax, df, filter_dict in all_df_list:
        # Plot scatter points with model-specific markers
        for model_name in df["model_name"].unique():
            data_subset = df[df["model_name"] == model_name]
            ax.scatter(
                data_subset["aut_score"],
                data_subset["metric_value"],
                label=model_name,
                color=model_name_to_color[model_name],
                marker=model_name_to_marker[model_name],
                alpha=0.9,
                s=80,
                edgecolors="black",
                linewidths=0.35,
            )

        # Add regression line
        sns.regplot(
            data=df,
            x="aut_score",
            y="metric_value",
            scatter=False,
            ax=ax,
            color="#8c6d46",
            line_kws={"linewidth": 2.0, "alpha": 0.95},
        )

        # Styling
        ax.set_facecolor("#f5f0e6")
        ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)

        # Title with model network and activation mode and the correlation coefficient and p value
        model_network = NETWORK_MAPPING.get(filter_dict["model_network"], filter_dict["model_network"])
        activation_mode = ACTIVATION_MODE_MAPPING.get(filter_dict["activation_mode"], filter_dict["activation_mode"])
        sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(filter_dict["model_data_sampling"], filter_dict["model_data_sampling"])
        corr_coef, p_value = pearsonr(df["aut_score"], df["metric_value"])
        ax.set_title(f"{activation_mode} (pooling: {sampling_label})\n(r={corr_coef:.2f}, p={p_value:.2f})", fontsize=15, pad=12)

        ax.set_xlabel("AUT Score")
        ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    brain_network = filters_lst[0]["brain_network"] if filters_lst else None
    fig.suptitle(f"Brain Alignment\n{BRAIN_NETWORK_MAPPING.get(brain_network, brain_network)} | {TASK_MAPPING.get(brain_network, brain_network)} | AUT Score", fontsize=17, y=0.98)

    # Create a shared, wrapped legend at the bottom
    legend_handles = [
        Line2D(
            [0], [0],
            marker=model_name_to_marker[model_name],
            linestyle="",
            markerfacecolor=model_name_to_color[model_name],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=10,
            color="none",
        )
        for model_name in sorted_models
    ]
    legend_labels = sorted_models
    legend_ncol = min(4, max(1, len(legend_labels)))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, 0.1),
        frameon=True,
        title="Model",
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")

def plot_gen_length_results(filters_lst, runs_lst, metric, gen_metric="mean", models=None, output_path=None):
    gen_len_df = pd.read_csv("experiments/templeton_aut/data/generation_length_stats_model_resp.csv")

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    all_model_names = set()
    all_df_list = []

    # Process each filter and collect data
    for idx, (filter_dict, runs) in enumerate(zip(filters_lst, runs_lst)):
        ax = axes[idx]

        # Extract data
        viz_data = []
        for run in runs:
            model_name = run.config["model_name"]
            if models and model_name not in models:
                continue
            
            gen_len_row = gen_len_df[gen_len_df["model_name"] == model_name]
            if gen_len_row.empty:
                print(f"No generation length data found for model '{model_name}'. Skipping.")
                continue
            gen_len = gen_len_row[gen_metric].values[0]
            model_size = extract_model_size(model_name)
            metric_value = get_dict_value(run.summary_metrics, metric)

            if model_size is None or metric_value is None:
                continue

            viz_data.append({
                "model_name": model_name,
                "generation_length": gen_len,
                "metric_value": metric_value,
            })
            all_model_names.add(model_name)

        if not viz_data:
            print(f"No valid data found for filter {idx}. Skipping.")
            continue

        df = pd.DataFrame(viz_data)
        df = df.sort_values(["generation_length", "model_name"])
        all_df_list.append((ax, df, filter_dict))

    # Determine color and marker mappings based on all unique model names
    sorted_models = sorted(all_model_names)
    palette = sns.color_palette("gist_earth", n_colors=max(len(sorted_models), 3))
    model_name_to_color = {name: palette[i] for i, name in enumerate(sorted_models)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    model_name_to_marker = {name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(sorted_models)}

    # Plot each subplot
    for ax, df, filter_dict in all_df_list:
        # Plot scatter points with model-specific markers
        for model_name in df["model_name"].unique():
            data_subset = df[df["model_name"] == model_name]
            ax.scatter(
                data_subset["generation_length"],
                data_subset["metric_value"],
                label=model_name,
                color=model_name_to_color[model_name],
                marker=model_name_to_marker[model_name],
                alpha=0.9,
                s=80,
                edgecolors="black",
                linewidths=0.35,
            )

        # Add regression line
        sns.regplot(
            data=df,
            x="generation_length",
            y="metric_value",
            scatter=False,
            ax=ax,
            color="#8c6d46",
            line_kws={"linewidth": 2.0, "alpha": 0.95},
        )

        # Styling
        ax.set_facecolor("#f5f0e6")
        ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)

        # Title with model network and activation mode and the correlation coefficient and p value
        model_network = NETWORK_MAPPING.get(filter_dict["model_network"], filter_dict["model_network"])
        activation_mode = ACTIVATION_MODE_MAPPING.get(filter_dict["activation_mode"], filter_dict["activation_mode"])
        sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(filter_dict["model_data_sampling"], filter_dict["model_data_sampling"])
        corr_coef, p_value = pearsonr(df["generation_length"], df["metric_value"])
        ax.set_title(f"{activation_mode} (pooling: {sampling_label})\n(r={corr_coef:.2f}, p={p_value:.2f})", fontsize=15, pad=12)

        ax.set_xlabel(f"{gen_metric.capitalize()} Generation Length")
        ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    fig.suptitle(f"{BRAIN_NETWORK_MAPPING.get(filters_lst[0]['brain_network'], filters_lst[0]['brain_network'])} | {TASK_MAPPING.get(filters_lst[0]['brain_network'], filters_lst[0]['brain_network'])} | {gen_metric.capitalize()} Generation Length", fontsize=17, y=0.98)

    # Create a shared, wrapped legend at the bottom
    legend_handles = [
        Line2D(
            [0], [0],
            marker=model_name_to_marker[model_name],
            linestyle="",
            markerfacecolor=model_name_to_color[model_name],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=10,
            color="none",
        )
        for model_name in sorted_models
    ]
    legend_labels = sorted_models
    legend_ncol = min(4, max(1, len(legend_labels)))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, 0.1),
        frameon=True,
        title="Model",
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")

def plot_main_results(aut_filter_dict, aut_runs, aut_short_filter_dict=None, aut_short_runs=None, 
                      metric=metric, models=None, output_path=None, plot_results=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    viz_data_model_size = []
    for run in aut_runs:
        model_name = run.config["model_name"]
        if models and model_name not in models:
            continue

        model_size = extract_model_size(model_name)
        metric_value = get_dict_value(run.summary_metrics, metric)
        if model_size is None or metric_value is None:
            continue

        viz_data_model_size.append(
            {
                "model_name": model_name,
                "model_size_b": model_size / 1e9,
                "metric_value": metric_value,
            }
        )

    viz_data_aut = []
    score_runs = aut_short_runs if aut_short_runs else aut_runs
    for run in score_runs:
        model_name = run.config["model_name"]
        if models and model_name not in models:
            continue

        metric_value = get_dict_value(run.summary_metrics, metric)
        if metric_value is None:
            continue

        aut_scoring_path = GEMINI_AUT_SCORING_PATHS.get(model_name)
        if aut_scoring_path is None:
            print(f"No Gemini AUT scoring path found for model '{model_name}'. Skipping AUT plot point.")
            continue
        else:
            aut_scoring_results = read_json(aut_scoring_path)
            if "aut_scoring" in aut_scoring_path:
                aut_score = np.nanmean([int(sample["output"].strip()[0]) for sample in aut_scoring_results["data"]])
            else:
                aut_score = np.nanmean([sample["gemini_aut_score"] for sample in aut_scoring_results["data"]])

        viz_data_aut.append(
            {
                "model_name": model_name,
                "aut_score": aut_score,
                "metric_value": metric_value,
            }
        )

    if not viz_data_model_size and not viz_data_aut:
        print("No valid data found for the selected filters. Skipping plot.")
        return

    df_model_size = pd.DataFrame(viz_data_model_size)
    if not df_model_size.empty:
        df_model_size = df_model_size.dropna(subset=["model_size_b", "metric_value"]).sort_values(["model_size_b", "model_name"])

    df_aut = pd.DataFrame(viz_data_aut)
    if not df_aut.empty:
        df_aut = df_aut.dropna(subset=["aut_score", "metric_value"]).sort_values(["aut_score", "model_name"])

    all_model_names = sorted(set(df_model_size.get("model_name", pd.Series(dtype=str)).tolist() + df_aut.get("model_name", pd.Series(dtype=str)).tolist()))
    if not all_model_names:
        print("No valid model entries remain after filtering. Skipping plot.")
        return

    if not plot_results:
        return df_model_size, df_aut

    palette = sns.color_palette("gist_earth", n_colors=max(len(all_model_names), 3))
    model_name_to_color = {name: palette[i] for i, name in enumerate(all_model_names)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    model_name_to_marker = {name: marker_cycle[i % len(marker_cycle)] for i, name in enumerate(all_model_names)}

    # Left subplot: model size
    ax = axes[0]
    for model_name in df_model_size["model_name"].unique():
        data_subset = df_model_size[df_model_size["model_name"] == model_name]
        ax.scatter(
            data_subset["model_size_b"],
            data_subset["metric_value"],
            color=model_name_to_color[model_name],
            marker=model_name_to_marker[model_name],
            alpha=0.9,
            s=80,
            edgecolors="black",
            linewidths=0.35,
        )
    if len(df_model_size) >= 2:
        sns.regplot(
            data=df_model_size,
            x="model_size_b",
            y="metric_value",
            scatter=False,
            ax=ax,
            color="#8c6d46",
            line_kws={"linewidth": 2.0, "alpha": 0.95},
        )
    ax.set_facecolor("#f5f0e6")
    ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}B"))
    model_size_sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(aut_filter_dict["model_data_sampling"], aut_filter_dict["model_data_sampling"])
    if len(df_model_size) >= 2:
        corr_coef, p_value = pearsonr(df_model_size["model_size_b"], df_model_size["metric_value"])
        ax.set_title(f"Model Size (pooling: {model_size_sampling_label})\n(r={corr_coef:.2f}, p={p_value:.2f})", fontsize=15, pad=12)
    else:
        ax.set_title(f"Model Size (pooling: {model_size_sampling_label})\n(insufficient data for correlation)", fontsize=15, pad=12)

    # Right subplot: Gemini AUT score
    ax = axes[1]
    for model_name in df_aut["model_name"].unique():
        data_subset = df_aut[df_aut["model_name"] == model_name]
        ax.scatter(
            data_subset["aut_score"],
            data_subset["metric_value"],
            color=model_name_to_color[model_name],
            marker=model_name_to_marker[model_name],
            alpha=0.9,
            s=80,
            edgecolors="black",
            linewidths=0.35,
        )
    if len(df_aut) >= 2:
        sns.regplot(
            data=df_aut,
            x="aut_score",
            y="metric_value",
            scatter=False,
            ax=ax,
            color="#8c6d46",
            line_kws={"linewidth": 2.0, "alpha": 0.95},
        )
    ax.set_facecolor("#f5f0e6")
    ax.grid(True, color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("AUT Score")
    ax.set_ylabel(LEGEND_MAPPING.get(metric, metric))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    aut_score_filter_dict = aut_short_filter_dict if aut_short_filter_dict else aut_filter_dict
    aut_score_sampling_label = MODEL_DATA_SAMPLING_MAPPING.get(aut_score_filter_dict["model_data_sampling"], aut_score_filter_dict["model_data_sampling"])
    if len(df_aut) >= 2:
        corr_coef, p_value = pearsonr(df_aut["aut_score"], df_aut["metric_value"])
        ax.set_title(f"AUT Score (pooling: {aut_score_sampling_label})\n(r={corr_coef:.2f}, p={p_value:.2f})", fontsize=15, pad=12)
    else:
        ax.set_title(f"AUT Score (pooling: {aut_score_sampling_label})\n(insufficient data for correlation)", fontsize=15, pad=12)

    model_network = NETWORK_MAPPING.get(aut_filter_dict["model_network"], aut_filter_dict["model_network"])
    activation_mode = ACTIVATION_MODE_MAPPING.get(aut_filter_dict["activation_mode"], aut_filter_dict["activation_mode"])
    fig.suptitle(f"Brain Alignment\n{BRAIN_NETWORK_MAPPING.get(aut_filter_dict['brain_network'], aut_filter_dict['brain_network'])} | {TASK_MAPPING.get(aut_filter_dict['brain_network'], aut_filter_dict['brain_network'])} | {activation_mode}", fontsize=17, y=0.98)

    legend_handles = [
        Line2D(
            [0], [0],
            marker=model_name_to_marker[model_name],
            linestyle="",
            markerfacecolor=model_name_to_color[model_name],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=10,
            color="none",
        )
        for model_name in all_model_names
    ]
    legend_ncol = min(4, max(1, len(all_model_names)))
    fig.legend(
        legend_handles,
        all_model_names,
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, 0.1),
        frameon=True,
        title="Model",
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")
    
    return df_model_size, df_aut


def plot_predictivity_barplot(runs, metric, models=None, title=None, output_path=None):
    viz_data = []

    for run in runs:
        model_name = run.config["model_name"]
        if models and model_name not in models:
            continue

        metric_value = get_dict_value(run.summary_metrics, metric)
        if metric_value is None:
            continue

        viz_data.append(
            {
                "model_name": model_name,
                "metric_value": metric_value,
            }
        )

    if not viz_data:
        print("No valid data found for the selected runs/models. Skipping plot.")
        return

    df = pd.DataFrame(viz_data)

    # Keep user-provided model order when available; otherwise sort by predictivity.
    if models:
        model_order = [m for m in models if m in df["model_name"].values]
        df["model_name"] = pd.Categorical(df["model_name"], categories=model_order, ordered=True)
        df = df.sort_values("model_name")
    else:
        df = df.sort_values("metric_value", ascending=True)

    palette = sns.color_palette("gist_earth", n_colors=max(len(df), 3))

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    sns.barplot(
        data=df,
        y="model_name",
        x="metric_value",
        palette=palette[: len(df)],
        ax=ax,
        orient="h",
        edgecolor="black",
        linewidth=0.35,
    )

    ax.set_facecolor("#f5f0e6")
    ax.grid(True, axis="x", color="#ddd2bf", linestyle="-", linewidth=0.7, alpha=0.8)
    x_label = LEGEND_MAPPING.get(metric, metric)
    ax.set_xlabel(str(x_label) if x_label is not None else "Predictivity")
    ax.set_ylabel("Model")
    ax.set_title(title or "Model Predictivity", fontsize=15, pad=12)

    # Add value labels at the end of each bar.
    max_val = df["metric_value"].max()
    x_offset = max(0.005, max_val * 0.01)
    for i, val in enumerate(df["metric_value"]):
        ax.text(val + x_offset, i, f"{val:.3f}", va="center", fontsize=11)

    plt.tight_layout()

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=360, bbox_inches="tight", facecolor="white")

    plt.show()


##########################################################################
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
    "allenai/Olmo-3-7B-Instruct"
]
models = [model.split("/")[-1].lower() for model in models]

# aut results
brain_networks = [
    ("yeo_dmn.*dt_create_with_ratings.json", "yeo_dmn"), 
    ("yeo_fp.*dt_create_with_ratings.json", "yeo_fp"),
    ("yeo_som.*dt_create.json", "yeo_som")
]
datasets = [
    (".*templeton_aut_create_eval_data.*", "aut"), 
    (".*templeton_aut_create_short_eval_data.*", "aut_short")
]
nc_thresholds = [0]
model_samplings = [("time:-1::layer:1:", "t_1"), ("time:mean::layer:1:", "t_mean")]

# for brain_network, brain_network_label in brain_networks:
#     for nc_th in nc_thresholds:
#         for activation_mode in ["prompt", "model_resp"]:
#             for (aut_model_sampling, aut_model_sampling_label), (aut_short_model_sampling, aut_short_model_sampling_label) in product(model_samplings, model_samplings):
#                 aut_filter_dict = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": ".*templeton_aut_create_eval_data.*",
#                         "model_data_sampling": aut_model_sampling,
#                         "network_pooling": None
#                 }
#                 aut_short_filter_dict = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": ".*templeton_aut_create_short_eval_data.*",
#                         "model_data_sampling": aut_short_model_sampling,
#                         "network_pooling": None
#                 }
#                 aut_runs = get_alignment_runs(**aut_filter_dict)
#                 aut_short_runs = get_alignment_runs(**aut_short_filter_dict)
#                 plot_main_results(
#                     aut_filter_dict,
#                     aut_runs,
#                     aut_short_filter_dict,
#                     aut_short_runs,
#                     metric,
#                     models=models,
#                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/main/main_results_{brain_network_label}_{activation_mode}_nc{nc_th}_{aut_model_sampling_label}_{aut_short_model_sampling_label}.pdf",
#                 )

# extended_models = models + [
#     "crpo-sft-llama-3.1-8b-instruct",
#     "crpo-dpo-llama-3.1-8b-instruct",
#     "deepseek-r1-distill-qwen-7b",
#     "qwen2.5-math-7b",
#     "qwen2.5-7b"
# ]
# #### individual plots for model size, aut score, and generation length
# for brain_network, brain_network_label in brain_networks:
#     for dataset, dataset_label in datasets:
#         for nc_th in nc_thresholds:
#             for model_sampling, model_sampling_label in model_samplings:
#                 filters_lst = [
#                     {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": "prompt",
#                         "dataset": dataset,
#                         "model_data_sampling": model_sampling,
#                         "network_pooling": None
#                     },
#                     {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": "model_resp",
#                         "dataset": dataset,
#                         "model_data_sampling": model_sampling,
#                         "network_pooling": None
#                     }
#                 ]
#                 runs_lst = [get_alignment_runs(**filter_dict) for filter_dict in filters_lst]
#                 print(f"Plotting results for brain network: {brain_network_label}, dataset: {dataset_label}, nc_th: {nc_th}, model_sampling: {model_sampling_label}")
#                 # plot_model_size_results(filters_lst, runs_lst, metric, models=models, 
#                                     #    output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/model_size_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")
#                 # plot_aut_score_results(filters_lst, runs_lst, metric, models=models,
#                                     #    output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/gemini_aut_score_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")
#                 plot_gen_length_results(filters_lst, runs_lst, metric, models=extended_models, gen_metric="mean",
#                                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/gen_mean_length_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")
#                 plot_gen_length_results(filters_lst, runs_lst, metric, models=extended_models, gen_metric="median",
#                                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/gen_median_length_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")

# for brain_network, brain_network_label in brain_networks:
#     for dataset, dataset_label in datasets:
#         for nc_th in nc_thresholds:
#             for model_sampling, model_sampling_label in model_samplings:
#                 for activation_mode in ["prompt", "model_resp"]:
#                     filter_dict = {
#                             "alignment_method": "rsa_per_subject",
#                             "nc_threshold": nc_th,
#                             "brain_network": brain_network,
#                             "model_network": "layers",
#                             "activation_mode": activation_mode,
#                             "dataset": dataset,
#                             "model_data_sampling": model_sampling,
#                             "network_pooling": None
#                         }
#                     runs = get_alignment_runs(**filter_dict)
#                     print(f"Plotting results for brain network: {brain_network_label}, dataset: {dataset_label}, nc_th: {nc_th}, model_sampling: {model_sampling_label}")
#                     # plot_model_size_results(filters_lst, runs_lst, metric, models=models, 
#                     #                         output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/model_size_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")
#                     # plot_aut_score_results(filters_lst, runs_lst, metric, models=models,
#                     #                        output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/gemini_aut_score_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")
#                     plot_best_layer_results(
#                         filter_dict,
#                         runs,
#                         metric,
#                         models=models,
#                         output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/best_layer_res_{brain_network_label}_{dataset_label}_{activation_mode}_nc{nc_th}_{model_sampling_label}.pdf",
#                     )

# for brain_network, brain_network_label in brain_networks:
#     for dataset, dataset_label in datasets:
#         for nc_th in nc_thresholds:
#             for activation_mode in ["prompt", "model_resp"]:
#                 filter_dict1 = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": dataset,
#                         "model_data_sampling": "time:-1::layer:1:",
#                         "network_pooling": None
#                 }
#                 filter_dict2 = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": dataset,
#                         "model_data_sampling": "time:mean::layer:1:",
#                         "network_pooling": None
#                 }
#                 runs1 = get_alignment_runs(**filter_dict1)
#                 runs2 = get_alignment_runs(**filter_dict2)
#                 print(f"Plotting results for brain network: {brain_network_label}, dataset: {dataset_label}, nc_th: {nc_th}")
#                 plot_best_layer_dist_comparison(
#                     filter_dict1,
#                     runs1,
#                     filter_dict2,
#                     runs2,
#                     metric,
#                     models=models,
#                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/best_layer_dist_res_{brain_network_label}_{dataset_label}_{activation_mode}_nc{nc_th}.pdf",
#                 )

#################################################

### oct results
for brain_network, brain_network_label in [("yeo_dmn.*dt_object.json", "yeo_dmn"), ("yeo_fp.*dt_object.json", "yeo_fp")]:
    for dataset, dataset_label in [(".*templeton_aut_object_eval_data.*", "oct")]:
        for nc_th in nc_thresholds:
            for model_sampling, model_sampling_label in [("time:-1::layer:1:", "t_1"), ("time:mean::layer:1:", "t_mean")]:
                filters_lst = [
                    {
                        "alignment_method": "rsa_per_subject",
                        "nc_threshold": nc_th,
                        "brain_network": brain_network,
                        "model_network": "layers",
                        "activation_mode": "prompt",
                        "dataset": dataset,
                        "model_data_sampling": model_sampling,
                        "network_pooling": None
                    },
                    {
                        "alignment_method": "rsa_per_subject",
                        "nc_threshold": nc_th,
                        "brain_network": brain_network,
                        "model_network": "layers",
                        "activation_mode": "model_resp",
                        "dataset": dataset,
                        "model_data_sampling": model_sampling,
                        "network_pooling": None
                    }
                ]
                runs_lst = [get_alignment_runs(**filter_dict, force_refresh=True) for filter_dict in filters_lst]
                print(f"Plotting results for brain network: {brain_network_label}, dataset: {dataset_label}, nc_th: {nc_th}, model_sampling: {model_sampling_label}")
                plot_model_size_results(filters_lst, runs_lst, metric, models=models,
                                        output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{dataset_label}/model_size_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")


# ######## empty prompt results
# brain_networks = [
#     ("yeo_dmn.*dt_create_with_ratings.json", "yeo_dmn")
# ]
# nc_thresholds = [0]

# for brain_network, brain_network_label in brain_networks:
#     for nc_th in nc_thresholds:
#         for activation_mode in ["prompt", "model_resp"]:
#             for (aut_model_sampling, aut_model_sampling_label) in model_samplings:
#                 aut_filter_dict = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": ".*templeton_aut_empty_eval_data.*",
#                         "model_data_sampling": aut_model_sampling,
#                         "network_pooling": None
#                 }
#                 aut_runs = get_alignment_runs(**aut_filter_dict, force_refresh=True)
#                 plot_main_results(
#                     aut_filter_dict,
#                     aut_runs,
#                     models=models,
#                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/no_prompt/no_prompt_results_{brain_network_label}_{activation_mode}_nc{nc_th}_{aut_model_sampling_label}.pdf",
#                 )

# # ######## non-lang prompt results
# brain_networks = [
#     ("yeo_dmn.*dt_create_with_ratings.json", "yeo_dmn")
# ]
# nc_thresholds = [0]

# for brain_network, brain_network_label in brain_networks:
#     for nc_th in nc_thresholds:
#         for activation_mode in ["prompt", "model_resp"]:
#             for (aut_model_sampling, aut_model_sampling_label) in model_samplings:
#                 aut_filter_dict = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": ".*templeton_aut_nolang_eval_data.*",
#                         "model_data_sampling": aut_model_sampling,
#                         "network_pooling": None
#                 }
#                 aut_runs = get_alignment_runs(**aut_filter_dict, force_refresh=True)
#                 plot_main_results(
#                     aut_filter_dict,
#                     aut_runs,
#                     models=models,
#                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/non_lang/non_lang_results_{brain_network_label}_{activation_mode}_nc{nc_th}_{aut_model_sampling_label}.pdf",
#                 )

## gemma results
# models = [
#     "google/gemma-3-270m-it",
#     "google/gemma-3-1b-it",
#     "google/gemma-3-4b-it",
#     "google/gemma-3-12b-it",
#     "google/gemma-3-27b-it"
# ]

## qwen results
# models = [
#     "Qwen/Qwen2.5-0.5B-Instruct",
#     "Qwen/Qwen2.5-1.5B-Instruct",
#     "Qwen/Qwen2.5-3B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct"
#     "Qwen/Qwen2.5-32B-Instruct",
#     "Qwen/Qwen2.5-72B-Instruct"
# ]

### llama results
# models = [
#     "meta-llama/Llama-3.2-1B-Instruct",
#     "meta-llama/Llama-3.2-3B-Instruct",
#     "meta-llama/Llama-3.1-70B-Instruct",
#     "meta-llama/Llama-3.1-8B",
# ]

# models = [model.split("/")[-1].lower() for model in models]
# for brain_network, brain_network_label in brain_networks:
#     for nc_th in nc_thresholds:
#         for activation_mode in ["prompt", "model_resp"]:
#             for (aut_model_sampling, aut_model_sampling_label), (aut_short_model_sampling, aut_short_model_sampling_label) in product(model_samplings, model_samplings):
#                 aut_filter_dict = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": ".*templeton_aut_create_eval_data.*",
#                         "model_data_sampling": aut_model_sampling,
#                         "network_pooling": None
#                 }
#                 aut_short_filter_dict = {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": activation_mode,
#                         "dataset": ".*templeton_aut_create_short_eval_data.*",
#                         "model_data_sampling": aut_short_model_sampling,
#                         "network_pooling": None
#                 }
#                 aut_runs = get_alignment_runs(**aut_filter_dict)
#                 aut_short_runs = get_alignment_runs(**aut_short_filter_dict)
#                 model_family = "gemma" if "gemma" in models[0].lower() else "qwen" if "qwen" in models[0].lower() else "llama" if "llama" in models[0].lower() else "other"
#                 plot_main_results(
#                     aut_filter_dict,
#                     aut_runs,
#                     aut_short_filter_dict,
#                     aut_short_runs,
#                     metric,
#                     models=models,
#                     output_path=f"experiments/templeton_aut/figures/{brain_network_label}/{model_family}/{model_family}_results_{brain_network_label}_{activation_mode}_nc{nc_th}_{aut_model_sampling_label}_{aut_short_model_sampling_label}.pdf"
#                 )

## print results
# aut_filter_dict = {
#         "alignment_method": "rsa_per_subject",
#         "nc_threshold": 0,
#         "brain_network": "yeo_dmn.*dt_create_with_ratings.json",
#         "model_network": "layers",
#         "activation_mode": "model_resp",
#         "dataset": ".*templeton_aut_create_eval_data.*",
#         "model_data_sampling": "time:-1::layer:1:",
#         "network_pooling": None
# }
# aut_short_filter_dict = {
#         "alignment_method": "rsa_per_subject",
#         "nc_threshold": 0,
#         "brain_network": "yeo_dmn.*dt_create_with_ratings.json",
#         "model_network": "layers",
#         "activation_mode": "model_resp",
#         "dataset": ".*templeton_aut_create_short_eval_data.*",
#         "model_data_sampling": "time:-1::layer:1:",
#         "network_pooling": None
# }
# aut_runs = get_alignment_runs(**aut_filter_dict)
# aut_short_runs = get_alignment_runs(**aut_short_filter_dict)
# results = plot_main_results(
#     aut_filter_dict,
#     aut_runs,
#     aut_short_filter_dict,
#     aut_short_runs,
#     metric,
#     models=models,
#     plot_results=False
# )
# if results and len(results) == 2:
#     df_model_size, df_aut = results
#     print("Model Size vs Predictivity:")
#     print(df_model_size[["model_name", "model_size_b", "metric_value"]])
#     print("\nAUT Score vs Predictivity:")
#     print(df_aut[["model_name", "aut_score", "metric_value"]])

################# yeo dmn aut vs oct comparison
# brain_networks = [
#     ("yeo_dmn.*dt_create_with_ratings.json", "yeo_dmn")
# ]
# datasets = [
#     (".*templeton_aut_object_eval_data.*", "oct"),
# ]

# for brain_network, brain_network_label in brain_networks:
#     for dataset, dataset_label in datasets:
#         for nc_th in nc_thresholds:
#             for model_sampling, model_sampling_label in model_samplings:
#                 filters_lst = [
#                     {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": "prompt",
#                         "dataset": dataset,
#                         "model_data_sampling": model_sampling,
#                         "network_pooling": None
#                     },
#                     {
#                         "alignment_method": "rsa_per_subject",
#                         "nc_threshold": nc_th,
#                         "brain_network": brain_network,
#                         "model_network": "layers",
#                         "activation_mode": "model_resp",
#                         "dataset": dataset,
#                         "model_data_sampling": model_sampling,
#                         "network_pooling": None
#                     }
#                 ]
#                 runs_lst = [get_alignment_runs(**filter_dict) for filter_dict in filters_lst]
#                 print(f"Plotting results for brain network: {brain_network_label}, dataset: {dataset_label}, nc_th: {nc_th}, model_sampling: {model_sampling_label}")
#                 plot_model_size_results(filters_lst, runs_lst, metric, models=models, 
#                                         output_path=f"experiments/templeton_aut/figures/{brain_network_label}_aut/{dataset_label}/model_size_res_{brain_network_label}_{dataset_label}_nc{nc_th}_{model_sampling_label}.pdf")