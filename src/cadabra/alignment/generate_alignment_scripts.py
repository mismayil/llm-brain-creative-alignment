import os
import hydra
from omegaconf import DictConfig

def get_nc_type(alignment_method):
    if "per_subject" in alignment_method:
        if "mean_voxel" in alignment_method:
            return "mean_voxel_per_subject"
        elif "median_voxel" in alignment_method:
            return "median_voxel_per_subject"
        elif "rsa" in alignment_method:
            return "rsa_per_subject"
    else:
        if "mean_voxel" in alignment_method:
            return "mean_voxel"
        elif "median_voxel" in alignment_method:
            return "median_voxel"
        elif "rsa" in alignment_method:
            return "rsa"
    return "per_voxel"

@hydra.main(version_base=None, config_path="configs", config_name="brain_alignment_script")
def main(config: DictConfig):
    alignment_data = config.alignment_data
    alignment_scripts = []
    nc_type = get_nc_type(config.alignment_method)
    nc_thresholds = ",".join(map(str, config.nc_thresholds))
    model_samplings = ",".join(config.model_sampling)
    brains = config.brains if len(config.brains) > 0 else [b["name"] for b in alignment_data["brain_data"]]
    models = config.models if len(config.models) > 0 else [m["name"] for m in alignment_data["model_data"]]

    for brain in brains:
        for model in models:
            for activation_mode in config.activation_modes:
                for network in config.model_networks:
                    brain_data = [b for b in alignment_data["brain_data"] if b["name"] == brain][0]
                    brain_datapath = brain_data["datapath"].replace("${brain_dir}", config.brain_dir)

                    model_data = [m for m in alignment_data["model_data"] if m["name"] == model][0]
                    model_datapaths = [d for d in model_data["datapaths"] if d["name"] == activation_mode]
                    
                    if len(model_datapaths) == 0:
                        print(f"Warning: No model datapath found for model {model} with path name {activation_mode}. Skipping.")
                        continue
                    
                    model_datapath_data = model_datapaths[0]
                    model_datapath = model_datapath_data["datapath"].replace("${model_dir}", config.model_dir)

                    nc_data = [nc for nc in brain_data["noise_ceilings"] if nc["name"] == nc_type][0]
                    nc_datapath = nc_data["datapath"].replace("${brain_dir}", config.brain_dir)

                    network_data_lst = [n for n in model_data.get("networks", []) if n["name"] == network]
                    network_data = network_data_lst[0] if len(network_data_lst) > 0 else None
                    network_datapath = network_data["datapath"].replace("${model_dir}", config.model_dir) if network_data else None

                    script = f"python -m cadabra.alignment.brain_alignment --multirun \\\n"
                    script += f"    alignment=\"{config.alignment_method}\" \\\n"
                    script += f"    model_args.model_datapath=\"{model_datapath}\" \\\n"
                    script += f"    brain_args.brain_datapath=\"{brain_datapath}\" \\\n"
                    script += f"    brain_args.noise_ceiling_path=\"{nc_datapath}\" \\\n"
                    script += f"    brain_args.noise_ceiling_threshold=\"{nc_thresholds}\" \\\n"
                    script += f"    model_args.model_data_sampling=\"{model_samplings}\" \\\n"
                    script += f"    wandb_project=\"{config.wandb_project}\" \\\n"

                    if network_data:
                        script += f"    model_args.model_network_path=\"{network_datapath}\" \\\n"
                        script += f"    model_args.model_network_type=\"network,random\"\n"
                    
                    alignment_scripts.append(script.strip("\\\n"))
    
    with open(config.output_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write("\n\n".join(alignment_scripts))
    
    os.chmod(config.output_path, 0o755)

    print(f"Generated {len(alignment_scripts)} alignment commands and saved to {config.output_path}")

if __name__ == "__main__":
    main()