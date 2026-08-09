import numpy as np
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv
from time import perf_counter

from cadabra.brain.fmri_utils import load_brain_neural_data, filter_dim_mismatch_samples
from cadabra.model.modeling_utils import load_model_neural_data
from cadabra.alignment.alignment_utils import ModelSample, BrainSample, load_noise_ceiling_results
from cadabra.utils import read_json, write_json, read_data, generate_datetime_id, prepare_metrics_for_wandb, wandb_log_run, get_stimuli_id, get_subject_id, get_wandb_runs, create_wandb_filters_from_config
from cadabra.alignment.sampling import BrainDataSampler, ModelDataSampler
from cadabra.alignment.alignment_methods import LinearRegressionAlignment, LinearRegressionPerSubjectAlignment, RSAAlignment, RSAPerSubjectAlignment

load_dotenv()

def compute_alignment(model_data, brain_data, method="linear", **alignment_kwargs):
    if method == "linear":
        aligner = LinearRegressionAlignment(**alignment_kwargs)
    elif method == "linear_per_subject":
        aligner = LinearRegressionPerSubjectAlignment(**alignment_kwargs)
    elif method == "rsa":
        aligner = RSAAlignment(**alignment_kwargs)
    elif method == "rsa_per_subject":
        aligner = RSAPerSubjectAlignment(**alignment_kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    alignment_results = aligner.compute(model_data, brain_data)
    return aligner, alignment_results

@hydra.main(version_base=None, config_path="configs", config_name="brain_alignment")
def main(config: DictConfig):
    start_time = perf_counter()

    # check if the run already exists in wandb
    if config.push_to_wandb:
        print("Checking for existing runs in Weights & Biases...")
        filters = create_wandb_filters_from_config(OmegaConf.to_container(config, resolve=True), key_prefix="config.config")
        existing_runs = get_wandb_runs(project=config.wandb_project, filters=filters)
        if existing_runs:
            print(f"Found {len(existing_runs)} existing run(s) with the same configuration:")
            for run in existing_runs:
                print(f"- Run ID: {run.id}, Name: {run.name}, URL: {run.url}")
            print("Exiting to avoid duplicate runs. If you want to run again, please change the configuration or delete the existing runs.")
            return
        else:
            print("No existing runs found with the same configuration. Proceeding with alignment computation.")
    
    print(f"Loading model data from {config.model_args.model_datapath}")
    print(f"Loading brain data from {config.brain_args.brain_datapath}")

    model_data = read_data(config.model_args.model_datapath)
    brain_data = read_data(config.brain_args.brain_datapath)

    # expecting one JSON file per data source
    if isinstance(model_data, list):
        if len(model_data) == 1:
            model_data = model_data[0]
        else:
            raise ValueError("Multiple model data files found. Please provide a single file.")
    
    if isinstance(brain_data, list):
        if len(brain_data) == 1:
            brain_data = brain_data[0]
        else:
            raise ValueError("Multiple brain data files found. Please provide a single file.")

    model_name = model_data["metadata"]["model_name"]
    dataset_name = brain_data["metadata"].get("dataset_name", brain_data["metadata"]["datapath"].split('/')[-1])

    noise_ceiling_results = load_noise_ceiling_results(config.brain_args.noise_ceiling_path, config.brain_args.noise_ceiling_threshold)
    ns_type = noise_ceiling_results["metadata"]["noise_ceiling_type"] if noise_ceiling_results is not None else None
    print(f"Noise ceiling type: {ns_type}")
    noise_ceiling_subjects = noise_ceiling_results["metadata"].get("subjects", []) if noise_ceiling_results else []
    if noise_ceiling_subjects:
        print(f"Number of noise ceiling valid subjects: {len(noise_ceiling_subjects)}")
    
    alignment_model_data = []
    alignment_brain_data = []

    model_sampler = ModelDataSampler(config.model_args.model_data_sampling, network_path=config.model_args.model_network_path, 
                                    network_type=config.model_args.model_network_type,
                                    ignore_first_layer=config.model_args.model_network_ignore_first_layer)
    brain_sampler = BrainDataSampler(config.brain_args.brain_data_sampling, noise_ceiling_data=noise_ceiling_results["data"] if noise_ceiling_results else None, 
                                     noise_ceiling_threshold=config.brain_args.noise_ceiling_threshold)

    for sample in tqdm(model_data["data"], desc="Preparing model data"):
        sample_datapath = sample.get(config.model_args.model_data_field, None)
        if sample_datapath is None:
            continue
        model_neural_data = load_model_neural_data(config.model_args.model_datapath, sample_datapath)
        model_neural_data = model_sampler.sample(model_neural_data)
        alignment_model_data.append(ModelSample(model_name=model_name, stimuli=get_stimuli_id(sample), data=model_neural_data, subject=get_subject_id(sample)))

    for sample in tqdm(brain_data["data"], desc="Loading brain neural data"):
        sample_datapath = sample.get(config.brain_args.brain_data_field, None)
        if sample_datapath is None:
            continue
        brain_neural_data = load_brain_neural_data(config.brain_args.brain_datapath, sample_datapath)
        sample[config.brain_args.brain_data_field] = brain_neural_data

    print("Filtering invalid brain samples...")
    print(f"Number of brain samples before filtering: {len(brain_data['data'])}")
    brain_data = filter_dim_mismatch_samples(brain_data["data"], data_field=config.brain_args.brain_data_field)
    print(f"Number of brain samples after filtering: {len(brain_data)}")

    print(f"Number of subjects in brain data: {len(set(get_subject_id(sample) for sample in brain_data))}")

    for sample in tqdm(brain_data, desc="Preparing brain neural data"):
        subject_id = get_subject_id(sample)
        if noise_ceiling_results and ns_type and "per_subject" in ns_type and subject_id not in noise_ceiling_subjects:
            # If the noise ceiling is per-subject and this subject is not in the noise ceiling results, we skip this sample
            continue
        brain_neural_data = brain_sampler.sample(sample[config.brain_args.brain_data_field])
        alignment_brain_data.append(BrainSample(subject=subject_id, stimuli=get_stimuli_id(sample), data=brain_neural_data))
    
    print(f"Number of subjects after noise ceiling filtering: {len(set(sample.subject for sample in alignment_brain_data))}")
    print(f"Prepared {len(alignment_model_data)} model samples and {len(alignment_brain_data)} brain samples for alignment.")

    print("Computing alignments...")
    alignment_args = OmegaConf.to_container(config.alignment.alignment_args, resolve=True)

    aligner, alignment_results = compute_alignment(alignment_model_data, alignment_brain_data, 
                                        method=config.alignment.alignment_method,
                                        **alignment_args)
    alignment_scores = alignment_results.alignment_scores
    
    model_datapath = pathlib.Path(config.model_args.model_datapath)
    output_dir = None
    unique_id = generate_datetime_id()

    if not config.output_dir:
        if model_datapath.is_file():
            output_dir = model_datapath.parent / "alignments" / unique_id
        else:
            output_dir = model_datapath / "alignments" / unique_id
    else:
        output_dir = pathlib.Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_basepath = f"alignment_{config.alignment.alignment_method}_mds_{config.model_args.model_data_sampling}_bds_{config.brain_args.brain_data_sampling}_{unique_id}.json".replace(":", "_")
    output_path = output_dir / output_basepath
    alignment_scores_path = output_path.with_suffix('.npy')
    np.save(alignment_scores_path, alignment_scores)
    print(f"Saved raw alignment scores to {alignment_scores_path}")

    subject_alignment_scores = alignment_results.subject_alignment_scores
    subject_alignment_scores_path = None

    if subject_alignment_scores is not None:
        subject_alignment_scores_path = output_path.with_name(f"{output_path.stem}_per_subject.npy")
        np.save(subject_alignment_scores_path, subject_alignment_scores)
        print(f"Saved per-subject alignment scores to {subject_alignment_scores_path}")

    alignment_metrics = []

    if noise_ceiling_results is not None:
        print("Applying noise ceiling adjustment to alignment results...")
        alignment_metrics = aligner.compute_metrics(alignment_scores)
        adjusted_alignment_scores = aligner.apply_noise_ceiling(alignment_results, noise_ceiling_results)
        adjusted_alignment_metrics = aligner.compute_metrics(adjusted_alignment_scores)
        for alignment_metric, adjusted_metric in zip(alignment_metrics, adjusted_alignment_metrics):
            alignment_metric["noise_ceiling_adjusted"] = adjusted_metric
        adjusted_alignment_scores_path = output_path.with_name(f"{output_path.stem}_ns_adjusted.npy")
        np.save(adjusted_alignment_scores_path, adjusted_alignment_scores)
        print(f"Saved noise-ceiling adjusted alignment scores to {adjusted_alignment_scores_path}")
    else:
        alignment_metrics = aligner.compute_metrics(alignment_scores)

    end_time = perf_counter()
    elapsed_time = end_time - start_time
    for alignment_metric in alignment_metrics:
        alignment_metric["alignment_time_seconds"] = elapsed_time
    print(f"Alignment computation time: {elapsed_time:.2f} seconds")

    output_data = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "config": OmegaConf.to_container(config, resolve=True),
            "output_path": str(output_path),
            "subjects": alignment_results.subjects
        }, 
        "metrics": alignment_metrics,
        "data": {
            "raw_alignment_scores_path": alignment_scores_path.name,
            "raw_per_subject_alignment_scores_path": subject_alignment_scores_path.name if subject_alignment_scores is not None else None,
            "noise_ceiling_adjusted_scores_path": adjusted_alignment_scores_path.name if noise_ceiling_results is not None else None
        }
    }

    print(f"Saving alignment results to {output_path}")
    write_json(output_data, output_path)

    if config.push_to_wandb:
        print("Logging results to Weights & Biases...")
        wandb_metrics = prepare_metrics_for_wandb(alignment_metrics)
        prompt_only = model_data["metadata"]["config"]["prompt_only"]
        model_sampling_parts = config.model_args.model_data_sampling.split("::")
        time_part = model_sampling_parts[0].replace("time:", "")
        model_sampling = "last" if time_part == "-1" else time_part
        model_network_suffix = ""
        if config.model_args.model_network_path:
            model_network_data = read_json(config.model_args.model_network_path)
            pooling = model_network_data["metadata"]["config"].get("pooling", "none")
            pooling = "".join([p[0] for p in pooling.split("-")])
            pct = model_network_data["metadata"]["config"].get("percentage", "0")
            model_network_suffix = f"-loc-{config.model_args.model_network_type}-pool-{pooling}-pct-{pct}"
        wandb_run_name = config.wandb_run_name or f"{model_name.lower()}-{'p' if prompt_only else 'r'}-{model_sampling}{model_network_suffix}"
        wandb_log_run(name=wandb_run_name, config=output_data["metadata"], metrics=wandb_metrics, project=config.wandb_project)

    print("All done.")

if __name__ == "__main__":
    main()