from tqdm import tqdm
import pathlib
from collections import defaultdict
from statistics import mean, median
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

load_dotenv()

from cadabra.utils import (
    read_json,
    write_json,
    find_files,
    compute_usage,
    wandb_log_run,
    prepare_metrics_for_wandb,
    detect_outliers_iqr,
)
from cadabra.model.modeling_utils import load_lm
from cadabra.metrics import (
    compute_content_frequency,
    compute_diversity,
    compute_novelty,
    compute_theme_uniqueness,
    compute_dsi,
    compute_n_gram_diversity,
    DEF_PREPROCESSING_ARGS,
    get_words,
    get_sentences,
    compute_dependency_complexity,
    compute_constituency_complexity,
    compute_flesch_readability_scores,
    compute_pos_complexity,
    compute_perplexity,
)


def remove_outliers(
    data,
    one_word_only=False,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
    iqr_threshold=1.5,
):
    new_data_ids = []

    lengths_in_words = []
    lengths_in_concepts = []

    for result in data:
        all_words = get_words(
            result["output"],
            lower=False,
            remove_punct=True,
            remove_stopwords=False,
            lemmatize=False,
            unique=False,
            dominant_k=None,
        )
        concepts = get_words(result["output"], **preprocessing_args)
        lengths_in_words.append(len(all_words))
        lengths_in_concepts.append(len(concepts))

    # remove outliers
    if not one_word_only:
        lengths_in_words_outliers = detect_outliers_iqr(
            lengths_in_words, multiplier=iqr_threshold
        )
    else:
        lengths_in_words_outliers = []

    for i, result in enumerate(data):
        if i not in lengths_in_words_outliers and lengths_in_concepts[i] > 1:
            new_data_ids.append(result["result_id"])

    new_data = [result for result in data if result["result_id"] in new_data_ids]
    outlier_data = [
        result for result in data if result["result_id"] not in new_data_ids
    ]

    return new_data, outlier_data


def compute_metrics(
    results,
    config,
    reference_model=None,
    reference_tokenizer=None,
):
    metrics = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    cost = {"input": 0, "output": 0, "total": 0}

    preprocessing_args = OmegaConf.to_container(config.preprocessing, resolve=True)

    data = [result for result in results["data"] if "output" in result]

    if config.num_samples and config.num_samples > 0:
        data = data[: config.num_samples]

    if preprocessing_args["remove_outliers"] or preprocessing_args.get("remove_1word_outliers_only"):
        print("Removing outliers...")
        data, outlier_data = remove_outliers(
            data,
            one_word_only=preprocessing_args.get("remove_1word_outliers_only"),
            preprocessing_args=preprocessing_args,
            iqr_threshold=preprocessing_args.get("iqr_threshold", 1.5),
        )

    def _compute_group_metrics(result_data):
        if len(result_data) > 1:
            texts = [res["output"] for res in result_data]
            diversity = compute_diversity(
                texts,
                config.embedding.model,
                config.embedding.type,
                config.embedding.strategy,
                config.embedding.distance_fn,
                preprocessing_args,
            )
            novelty = compute_novelty(
                texts,
                config.embedding.model,
                config.embedding.type,
                config.embedding.distance_fn,
                preprocessing_args,
            )
            theme_uniq, theme_clusters = compute_theme_uniqueness(
                texts,
                config.embedding.model,
                config.embedding.type,
                config.embedding.strategy,
                config.clustering.linkage,
                config.clustering.dist_threshold,
                preprocessing_args,
            )
            corpus_n_gram_diversity, corpus_n_gram_frequency = (
                compute_n_gram_diversity(" ".join(texts), config.max_n_gram)
            )

            return {
                "diversity": diversity,
                "novelty": novelty,
                "theme_uniq": theme_uniq,
                "theme_clusters": theme_clusters,
                "corpus_n_gram_diversity": corpus_n_gram_diversity,
                "corpus_n_gram_frequency": corpus_n_gram_frequency,
            }

    # compute metrics per prompt
    data_by_id = defaultdict(list)
    for result in data:
        data_by_id[result["id"]].append(result)

    for _, group_data in tqdm(
        data_by_id.items(), desc="Computing group metrics"
    ):
        if len(group_data) > 1:
            group_metrics = _compute_group_metrics(group_data)
            for (
                group_result,
                diversity_score,
                novelty_score,
                theme_uniq_score,
                theme_cluster,
            ) in zip(
                group_data,
                group_metrics["diversity"],
                group_metrics["novelty"],
                group_metrics["theme_uniq"],
                group_metrics["theme_clusters"],
            ):
                if "metrics" not in group_result:
                    group_result["metrics"] = {}
                group_result["metrics"]["diversity"] = diversity_score
                group_result["metrics"]["novelty"] = novelty_score
                group_result["metrics"]["theme_uniq"] = theme_uniq_score
                group_result["metrics"]["theme_cluster"] = theme_cluster

    for result in tqdm(data, desc="Computing individual metrics"):
        if "metrics" not in result:
            result["metrics"] = {}

        all_words = get_words(
            result["output"],
            lower=False,
            remove_punct=True,
            remove_stopwords=False,
            lemmatize=False,
            unique=False,
            dominant_k=None,
        )
        unique_words = list(set([word.lower() for word in all_words if all_words]))
        concepts = get_words(
            result["output"],
            lower=True,
            remove_punct=True,
            remove_stopwords=True,
            lemmatize=True,
            unique=True,
            dominant_k=None,
        )
        sentences = get_sentences(result["output"])
        sentence_words = [
            get_words(
                sentence,
                lower=False,
                remove_punct=True,
                remove_stopwords=False,
                lemmatize=False,
                unique=False,
                dominant_k=None,
            )
            for sentence in sentences
        ]
        sentence_unique_words = [
            list(set([word.lower() for word in words])) for words in sentence_words
        ]

        # basic metrics
        result["metrics"]["length_in_chars"] = len(result["output"])
        result["metrics"]["length_in_words"] = len(all_words)
        result["metrics"]["length_in_unique_words"] = len(unique_words)
        result["metrics"]["length_in_concepts"] = len(concepts)
        result["metrics"]["type_token_ratio"] = (
            len(unique_words) / len(all_words) if all_words else 0
        )
        result["metrics"]["mean_word_length_in_chars"] = mean(
            [len(word) for word in all_words]
        )

        result["metrics"]["length_in_sentences"] = len(sentences)
        result["metrics"]["mean_sentence_length_in_chars"] = mean(
            [len(sentence) for sentence in sentences]
        )
        result["metrics"]["mean_sentence_length_in_words"] = mean(
            [len(words) for words in sentence_words]
        )
        result["metrics"]["mean_sentence_length_in_unique_words"] = mean(
            [len(words) for words in sentence_unique_words]
        )
        result["metrics"]["length_in_first_person_singular"] = len(
            [
                word
                for word in all_words
                if word.lower() in ["i", "me", "my", "mine", "myself"]
            ]
        )
        result["metrics"]["length_in_first_person_plural"] = len(
            [
                word
                for word in all_words
                if word.lower() in ["we", "us", "our", "ours", "ourselves"]
            ]
        )
        result["metrics"]["length_in_second_person"] = len(
            [
                word
                for word in all_words
                if word.lower()
                in ["you", "your", "yours", "yourself", "yourselves"]
            ]
        )
        result["metrics"]["length_in_third_person_singular"] = len(
            [
                word
                for word in all_words
                if word.lower()
                in [
                    "he",
                    "him",
                    "his",
                    "himself",
                    "she",
                    "her",
                    "hers",
                    "herself",
                    "it",
                    "its",
                    "itself",
                ]
            ]
        )
        result["metrics"]["length_in_third_person_plural"] = len(
            [
                word
                for word in all_words
                if word.lower() in ["they", "them", "their", "theirs", "themselves"]
            ]
        )
        result["metrics"]["length_in_first_person"] = (
            result["metrics"]["length_in_first_person_singular"]
            + result["metrics"]["length_in_first_person_plural"]
        )
        result["metrics"]["length_in_third_person"] = (
            result["metrics"]["length_in_third_person_singular"]
            + result["metrics"]["length_in_third_person_plural"]
        )

        # complex metrics
        result["metrics"]["dsi"] = compute_dsi(
            result["output"],
            config.embedding.model,
            config.embedding.type,
            config.embedding.distance_fn,
            preprocessing_args,
        )
        result["metrics"]["n_gram_diversity"], _ = compute_n_gram_diversity(
            result["output"],
            config.max_n_gram,
        )

        dependency_paths, dependency_num_clauses = compute_dependency_complexity(
            result["output"]
        )
        result["metrics"]["mean_dep_num_clauses"] = mean(dependency_num_clauses)
        result["metrics"]["max_dep_num_clauses"] = max(dependency_num_clauses)
        result["metrics"]["mean_dep_path_length"] = mean(
            [
                mean([len(path) for path, freq in path_counter.items()])
                for path_counter in dependency_paths
            ]
        )
        result["metrics"]["max_dep_path_length"] = max(
            [
                max([len(path) for path, freq in path_counter.items()])
                for path_counter in dependency_paths
            ]
        )

        constituency_complexity = compute_constituency_complexity(result["output"])
        result["metrics"]["mean_constituency_tree_depth"] = mean(
            constituency_complexity
        )
        result["metrics"]["max_constituency_tree_depth"] = max(
            constituency_complexity
        )

        flesch_ease, flesch_kincaid = compute_flesch_readability_scores(
            result["output"]
        )
        result["metrics"]["readability_flesch_ease"] = flesch_ease
        result["metrics"]["readability_flesch_kincaid"] = flesch_kincaid

        pos_complexity = compute_pos_complexity(result["output"])
        for pos, pos_comps in pos_complexity.items():
            result["metrics"][f"mean_pos_{pos.lower()}_ratio"] = (
                mean(pos_comps) if pos_comps else 0
            )
        content_frequency = compute_content_frequency(
            result["output"], corpus_index=config.corpus_index
        )
        result["metrics"]["content_frequency"] = content_frequency
        norm_freqs = [norm_freq for token, pos, freq, norm_freq in content_frequency]
        result["metrics"]["mean_norm_content_freq"] = mean(norm_freqs)
        result["metrics"]["median_norm_content_freq"] = median(norm_freqs)
        result["metrics"]["max_norm_content_freq"] = max(norm_freqs)
        result["metrics"]["min_norm_content_freq"] = min(norm_freqs)
        
        # for adjectives
        adj_freqs = [
            norm_freq
            for token, pos, freq, norm_freq in content_frequency
            if pos == "ADJ"
        ]
        if adj_freqs:
            result["metrics"]["mean_norm_adj_content_freq"] = mean(adj_freqs)
            result["metrics"]["median_norm_adj_content_freq"] = median(adj_freqs)
            result["metrics"]["max_norm_adj_content_freq"] = max(adj_freqs)
            result["metrics"]["min_norm_adj_content_freq"] = min(adj_freqs)

        if config.report_usage:
            sample_usage, sample_cost = compute_usage(
                result, results["metadata"]["config"]["model"]
            )

            if sample_usage:
                usage["input_tokens"] += sample_usage["input_tokens"]
                usage["output_tokens"] += sample_usage["output_tokens"]
                usage["total_tokens"] += (
                    sample_usage["input_tokens"] + sample_usage["output_tokens"]
                )

            if sample_cost:
                cost["input"] += sample_cost["input"]
                cost["output"] += sample_cost["output"]
                cost["total"] += sample_cost["total"]

    if reference_model:
        is_instruct_model = (
            "instruct" in config.reference_model.lower()
            or "-it" in config.reference_model.lower()
        )
        if is_instruct_model:
            perplexity_data = [
                [
                    {"role": "user", "content": result["user_prompt"][0] if isinstance(result["user_prompt"], list) else result["user_prompt"]},
                    {"role": "assistant", "content": result["output"]},
                ]
                for result in data
            ]
        else:
            perplexity_data = [
                f'{result["user_prompt"][0]}\n{result["output"]}' if isinstance(result["user_prompt"], list) else f'{result["user_prompt"]}\n{result["output"]}' for result in data
            ]
        perplexities = compute_perplexity(
            perplexity_data,
            reference_model,
            reference_tokenizer,
            batch_size=config.batch_size,
        )
        for result, perplexity in zip(data, perplexities):
            result["metrics"]["perplexity"] = perplexity
        metrics["mean_perplexity"] = mean(perplexities)

    def _aggregate_metrics(metric_data):
        # basic metrics
        agg_metrics = {}
        agg_metrics["mean_length_in_chars"] = mean(
            [result["metrics"]["length_in_chars"] for result in metric_data]
        )
        agg_metrics["median_length_in_chars"] = median(
            [result["metrics"]["length_in_chars"] for result in metric_data]
        )
        agg_metrics["mean_length_in_words"] = mean(
            [result["metrics"]["length_in_words"] for result in metric_data]
        )
        agg_metrics["median_length_in_words"] = median(
            [result["metrics"]["length_in_words"] for result in metric_data]
        )
        agg_metrics["mean_length_in_unique_words"] = mean(
            [result["metrics"]["length_in_unique_words"] for result in metric_data]
        )
        agg_metrics["median_length_in_unique_words"] = median(
            [result["metrics"]["length_in_unique_words"] for result in metric_data]
        )
        agg_metrics["mean_length_in_concepts"] = mean(
            [result["metrics"]["length_in_concepts"] for result in metric_data]
        )
        agg_metrics["median_length_in_concepts"] = median(
            [result["metrics"]["length_in_concepts"] for result in metric_data]
        )

        agg_metrics["mean_word_length_in_chars"] = mean(
            [result["metrics"]["mean_word_length_in_chars"] for result in metric_data]
        )

        agg_metrics["mean_length_in_sentences"] = mean(
            [result["metrics"]["length_in_sentences"] for result in metric_data]
        )
        agg_metrics["median_length_in_sentences"] = median(
            [result["metrics"]["length_in_sentences"] for result in metric_data]
        )
        agg_metrics["mean_sentence_length_in_chars"] = mean(
            [
                result["metrics"]["mean_sentence_length_in_chars"]
                for result in metric_data
            ]
        )
        agg_metrics["mean_sentence_length_in_words"] = mean(
            [
                result["metrics"]["mean_sentence_length_in_words"]
                for result in metric_data
            ]
        )
        agg_metrics["mean_sentence_length_in_unique_words"] = mean(
            [
                result["metrics"]["mean_sentence_length_in_unique_words"]
                for result in metric_data
            ]
        )

        # complex metrics
        diversity = [
            result["metrics"]["diversity"]
            for result in metric_data
            if "diversity" in result["metrics"]
        ]
        if diversity:
            agg_metrics["mean_diversity"] = mean(diversity)
            agg_metrics["median_diversity"] = median(
                diversity
            )

        novelty = [
            result["metrics"]["novelty"]
            for result in metric_data
            if "novelty" in result["metrics"]
        ]
        if novelty:
            agg_metrics["mean_novelty"] = mean(novelty)
            agg_metrics["median_novelty"] = median(novelty)

            metric_data_by_id = defaultdict(list)
            for result in metric_data:
                if "novelty" in result["metrics"]:
                    metric_data_by_id[result["id"]].append(
                        result["metrics"]["novelty"]
                    )

            # sort items
            for key in metric_data_by_id:
                metric_data_by_id[key].sort(reverse=True)

            agg_metrics["top1_novelty"] = mean(
                [novelties[0] for key, novelties in metric_data_by_id.items()]
            )
            agg_metrics["top3_novelty"] = mean(
                [mean(novelties[:3]) for key, novelties in metric_data_by_id.items()]
            )
            agg_metrics["top5_novelty"] = mean(
                [mean(novelties[:5]) for key, novelties in metric_data_by_id.items()]
            )

        theme_uniq = [
            result["metrics"]["theme_uniq"]
            for result in metric_data
            if "theme_uniq" in result["metrics"]
        ]
        if theme_uniq:
            agg_metrics["mean_theme_uniq"] = mean(
                theme_uniq
            )
            agg_metrics["median_theme_uniq"] = median(
                theme_uniq
            )

        agg_metrics["mean_dsi"] = mean(
            [result["metrics"]["dsi"] for result in metric_data]
        )
        agg_metrics["median_dsi"] = median(
            [result["metrics"]["dsi"] for result in metric_data]
        )

        agg_metrics["mean_n_gram_diversity"] = []
        for n_gram_len in range(1, config.max_n_gram + 1):
            n_gram_diversity = [
                result["metrics"]["n_gram_diversity"][n_gram_len - 1]
                for result in metric_data
                if len(result["metrics"]["n_gram_diversity"]) >= n_gram_len
            ]
            if n_gram_diversity:
                agg_metrics["mean_n_gram_diversity"].append(mean(n_gram_diversity))

        agg_metrics["mean_dep_num_clauses"] = mean(
            [result["metrics"]["mean_dep_num_clauses"] for result in metric_data]
        )
        agg_metrics["mean_max_dep_num_clauses"] = mean(
            [result["metrics"]["max_dep_num_clauses"] for result in metric_data]
        )
        agg_metrics["mean_dep_path_length"] = mean(
            [result["metrics"]["mean_dep_path_length"] for result in metric_data]
        )
        agg_metrics["mean_max_dep_path_length"] = mean(
            [result["metrics"]["max_dep_path_length"] for result in metric_data]
        )
        perplexity = [
            result["metrics"]["perplexity"]
            for result in metric_data
            if "perplexity" in result["metrics"]
        ]
        if perplexity:
            agg_metrics["mean_perplexity"] = mean(perplexity)
            agg_metrics["median_perplexity"] = median(perplexity)

        agg_metrics["mean_norm_content_freq"] = mean(
            [result["metrics"]["mean_norm_content_freq"] for result in metric_data]
        )
        agg_metrics["median_norm_content_freq"] = median(
            [result["metrics"]["median_norm_content_freq"] for result in metric_data]
        )

        mean_norm_adj_content_freqs = [
            result["metrics"]["mean_norm_adj_content_freq"]
            for result in metric_data
            if "mean_norm_adj_content_freq" in result["metrics"]
        ]
        median_norm_adj_content_freqs = [
            result["metrics"]["median_norm_adj_content_freq"]
            for result in metric_data
            if "median_norm_adj_content_freq" in result["metrics"]
        ]
        agg_metrics["mean_norm_adj_content_freq"] = mean(mean_norm_adj_content_freqs)
        agg_metrics["median_norm_adj_content_freq"] = median(median_norm_adj_content_freqs)
    
        return agg_metrics

    metrics.update(_aggregate_metrics(data))

    metrics["usage"] = usage
    metrics["cost"] = cost
    metrics["num_total_samples"] = len(results["data"])
    metrics["num_valid_samples"] = len(data)

    return metrics


def report_metrics(
    results_files,
    config=None,
    reference_model=None,
    reference_tokenizer=None,
):
    for results_file in results_files:
        results = read_json(results_file)

        try:
            if "data" in results:
                print(f"Reporting metrics for: {results_file}")
                metrics = compute_metrics(
                    results,
                    config=config,
                    reference_model=reference_model,
                    reference_tokenizer=reference_tokenizer,
                )
                results["metadata"]["eval_config"] = OmegaConf.to_container(config, resolve=True)
                results["metrics"] = metrics

                output_file = results_file

                if config.output_dir:
                    # If output_dir is specified, save the results in that directory
                    output_dir = pathlib.Path(config.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    dataset_name = f"{pathlib.Path(results_file).stem}"
                    output_file = output_dir / f"{dataset_name}.json"
                    results["metadata"]["dataset"] = dataset_name
                    results["metadata"]["eval_wandb_run_id"] = None

                if config.wandb.project:
                    run_metrics = prepare_metrics_for_wandb(
                        metrics, exclude_prefixes=["num_", "usage", "cost"]
                    )
                    metadata = results["metadata"]
                    model_name = metadata.get("model_name", metadata["config"]["model_path"].split("/")[-1])
                    previous_run_id = metadata.get("eval_wandb_run_id")
                    dataset = metadata.get("dataset")

                    if not dataset:
                        dataset = metadata["config"]["data_path"].split("/")[-1]
                        metadata["dataset"] = dataset

                    wandb_run = wandb_log_run(
                        name=model_name,
                        project=config.wandb.project,
                        metrics=run_metrics,
                        config=metadata,
                        run_id=previous_run_id,
                    )
                    metadata["eval_wandb_run_id"] = wandb_run.id

                write_json(results, output_file)
        except Exception as e:
            print(results_file)
            raise e


def evaluate_lm(config: DictConfig):
    files_to_process = []

    results_path = pathlib.Path(config.results_path)

    if results_path.is_file():
        files_to_process.append(config.results_path)
    else:
        files_to_process.extend(find_files(config.results_path, extension="json"))

    if not files_to_process:
        print(f"No files found in {config.results_path}")
        return

    reference_model, reference_tokenizer = (
        load_lm(config.reference_model)
        if config.reference_model
        else (None, None)
    )

    report_metrics(
        files_to_process,
        config=config,
        reference_model=reference_model,
        reference_tokenizer=reference_tokenizer,
    )

@hydra.main(version_base=None, config_path="configs", config_name="evaluate_lm")
def main(config: DictConfig):
    evaluate_lm(config)

if __name__ == "__main__":
    main()