#!/bin/bash

models=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "CNCL-Penn-State/CrPO-llama-3.1-8b-instruct-cre"
    # "google/gemma-3-270m-it"
    # "google/gemma-3-1b-it"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-27b-it"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "allenai/Olmo-3.1-32B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.1-8B"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "tiiuae/falcon-40b-instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "Qwen/Qwen2.5-Math-7B"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "CNCL-Penn-State/CrPO-sft-llama-3.1-8b-instruct"
    # "CNCL-Penn-State/CrPO-dpo-llama-3.1-8b-instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "allenai/Olmo-3-7B-Instruct"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
)

# # AUT task
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python -m cadabra.model.inference_lm \
#         model_path="$model" \
#         data_path="experiments/templeton_aut/data/eval/templeton_aut_create_eval_data.json" \
#         output_dir="../outputs/templeton_aut/aut/$model_name" \
#         gen_args="sampling_t0.7_p0.95" \
#         gen_args.max_new_tokens=1024
# done

# # OCT task
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python -m cadabra.model.inference_lm \
#         model_path="$model" \
#         data_path="experiments/templeton_aut/data/eval/templeton_aut_object_eval_data.json" \
#         output_dir="../outputs/templeton_aut/oct/$model_name" \
#         gen_args="sampling_t0.7_p0.95" \
#         gen_args.max_new_tokens=1024
# done

# # AUT task with short prompt
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python -m cadabra.model.inference_lm \
#         model_path="$model" \
#         data_path="experiments/templeton_aut/data/eval/templeton_aut_create_short_eval_data.json" \
#         output_dir="../outputs/templeton_aut/aut/$model_name" \
#         gen_args="sampling_t0.7_p0.95" \
#         gen_args.max_new_tokens=1024
# done

# # OCT task with short prompt
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python -m cadabra.model.inference_lm \
#         model_path="$model" \
#         data_path="experiments/templeton_aut/data/eval/templeton_aut_object_short_eval_data.json" \
#         output_dir="../outputs/templeton_aut/oct/$model_name" \
#         gen_args="sampling_t0.7_p0.95" \
#         gen_args.max_new_tokens=1024
# done

# AUT task with forced responses
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python -m cadabra.model.inference_lm \
#         model_path="$model" \
#         data_path="experiments/templeton_aut/data/eval/templeton_aut_create_eval_data_with_subjects.json" \
#         output_dir="../outputs/templeton_aut/aut/$model_name" \
#         force_response=True
# done

# Empty task
for model in "${models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    python -m cadabra.model.inference_lm \
        model_path="$model" \
        data_path="experiments/templeton_aut/data/eval/templeton_aut_empty_eval_data.json" \
        output_dir="../outputs/templeton_aut/aut/$model_name" \
        gen_args="sampling_t0.7_p0.95" \
        gen_args.max_new_tokens=1024
done

# No language task
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     python -m cadabra.model.inference_lm \
#         model_path="$model" \
#         data_path="experiments/templeton_aut/data/eval/templeton_aut_nolang_eval_data.json" \
#         output_dir="../outputs/templeton_aut/aut/$model_name" \
#         gen_args="sampling_t0.7_p0.95" \
#         gen_args.max_new_tokens=1024
# done

########################################## network and random ablation experiments ##########################################
# model_dir="../outputs/templeton_aut"
# models_with_paths=(
    # "meta-llama/Llama-3.1-8B-Instruct:${model_dir}/localizations/llama-3.1-8b-instruct/20251216_202450/localization_20251216_202450_r100-100_plast-token_u1351_pct1.json"
    # "CNCL-Penn-State/CrPO-llama-3.1-8b-instruct-cre:${model_dir}/localizations/crpo-llama-3.1-8b-instruct-cre/20251216_215809/localization_20251216_215809_r100-100_plast-token_u1351_pct1.json"
    # "google/gemma-3-270m-it:${model_dir}/localizations/gemma-3-270m-it/20260318_004241/localization_20260318_004241_r100-100_plast-token_u121_pct1.json"
    # "google/gemma-3-1b-it:${model_dir}/localizations/gemma-3-1b-it/20260318_004318/localization_20260318_004318_r100-100_plast-token_u311_pct1.json"
    # "google/gemma-3-4b-it:${model_dir}/localizations/gemma-3-4b-it/20260318_004412/localization_20260318_004412_r100-100_plast-token_u896_pct1.json"
    # "google/gemma-3-12b-it:${model_dir}/localizations/gemma-3-12b-it/20260318_004548/localization_20260318_004548_r100-100_plast-token_u1882_pct1.json"
    # "google/gemma-3-27b-it:${model_dir}/localizations/gemma-3-27b-it/20260318_004853/localization_20260318_004853_r100-100_plast-token_u3391_pct1.json"
    # "meta-llama/Llama-3.2-1B-Instruct:${model_dir}/localizations/llama-3.2-1b-instruct/20260318_005402/localization_20260318_005402_r100-100_plast-token_u348_pct1.json"
    # "meta-llama/Llama-3.2-3B-Instruct:${model_dir}/localizations/llama-3.2-3b-instruct/20260318_005500/localization_20260318_005500_r100-100_plast-token_u890_pct1.json"
    # "allenai/Olmo-3.1-32B-Instruct:${model_dir}/localizations/olmo-3.1-32b-instruct/20260322_181159/localization_20260322_181159_r100-100_plast-token_u3329_pct1.json"
    # "meta-llama/Llama-3.1-70B-Instruct:${model_dir}/localizations/llama-3.1-70b-instruct/20260322_181557/localization_20260322_181557_r100-100_plast-token_u6643_pct1.json"
# )

# for model_with_path in "${models_with_paths[@]}"; do
#     IFS=":" read -r model ablation_path <<< "$model_with_path"
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

#     # network ablation
#     # python -m cadabra.model.inference_lm \
#     #     model_path="$model" \
#     #     data_path="experiments/templeton_aut/data/eval/templeton_aut_create_short_eval_data.json" \
#     #     output_dir="../outputs/templeton_aut/aut/${model_name}" \
#     #     gen_args="sampling_t0.7_p0.95" \
#     #     gen_args.max_new_tokens=1024 \
#     #     ablation.type="network" \
#     #     ablation.path="$ablation_path"

#     # random ablation
#     # python -m cadabra.model.inference_lm --multirun \
#     #     model_path="$model" \
#     #     data_path="experiments/templeton_aut/data/eval/templeton_aut_create_short_eval_data.json" \
#     #     output_dir="../outputs/templeton_aut/aut/${model_name}" \
#     #     gen_args="sampling_t0.7_p0.95" \
#     #     gen_args.max_new_tokens=1024 \
#     #     ablation.type="random" \
#     #     ablation.path="$ablation_path" \
#     #     seed=328,1204
# done
