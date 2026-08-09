#!/bin/bash

input_paths=(
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260306_140725/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260306_140725.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260306_140819/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260306_140819.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260316_153058/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260316_153058.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260316_153639/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260316_153639.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260316_154445/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260316_154445.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260316_155546/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260316_155546.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260316_161156/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260316_161156.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260316_163439/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260316_163439.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260316_163502/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260316_163502.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260322_145141/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260322_145141.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260322_145227/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260322_145227.json"
       # "../outputs/templeton_aut/aut/mistral-7b-instruct-v0.3/20260325_180953/templeton_aut_create_short_eval_data_mistral-7b-instruct-v0.3_20260325_180953.json"
       # "../outputs/templeton_aut/aut/crpo-mistral-7b-instruct-cre/20260325_181021/templeton_aut_create_short_eval_data_crpo-mistral-7b-instruct-cre_20260325_181021.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b/20260327_200542/templeton_aut_create_short_eval_data_llama-3.1-8b_20260327_200542.json"
       # "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260327_201250/templeton_aut_create_short_eval_data_llama-3.1-minitaur-8b_20260327_201250.json"
       # "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260328_210352/templeton_aut_create_short_eval_data_qwen2.5-32b-instruct_20260328_210352.json"
       # "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260328_210430/templeton_aut_create_short_eval_data_qwen2.5-72b-instruct_20260328_210430.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260328_210527/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-8b_20260328_210527.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260328_211319/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-70b_20260328_211319.json"
       # "../outputs/templeton_aut/aut/falcon-40b-instruct/20260328_214322/templeton_aut_create_short_eval_data_falcon-40b-instruct_20260328_214322.json"
       # "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260328_214636/templeton_aut_create_short_eval_data_qwen2.5-14b-instruct_20260328_214636.json"

       # "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260527_111143/templeton_aut_create_short_eval_data_qwen2.5-7b-instruct_20260527_111143.json"
       # "../outputs/templeton_aut/aut/olmo-3-7b-instruct/20260801_100207/templeton_aut_create_short_eval_data_olmo-3-7b-instruct_20260801_100207.json"
       # "../outputs/templeton_aut/aut/qwen2.5-0.5b-instruct/20260801_161344/templeton_aut_create_short_eval_data_qwen2.5-0.5b-instruct_20260801_161344.json"
       # "../outputs/templeton_aut/aut/qwen2.5-1.5b-instruct/20260801_161435/templeton_aut_create_short_eval_data_qwen2.5-1.5b-instruct_20260801_161435.json"
       # "../outputs/templeton_aut/aut/qwen2.5-3b-instruct/20260801_161453/templeton_aut_create_short_eval_data_qwen2.5-3b-instruct_20260801_161453.json"
       "../outputs/templeton_aut/aut/deepseek-r1-distill-qwen-7b/20260527_105455/templeton_aut_create_short_eval_data_deepseek-r1-distill-qwen-7b_20260527_105455.json"
       "../outputs/templeton_aut/aut/qwen2.5-math-7b/20260527_110045/templeton_aut_create_short_eval_data_qwen2.5-math-7b_20260527_110045.json"
       "../outputs/templeton_aut/aut/qwen2.5-7b/20260527_110707/templeton_aut_create_short_eval_data_qwen2.5-7b_20260527_110707.json"
       "../outputs/templeton_aut/aut/crpo-sft-llama-3.1-8b-instruct/20260527_111159/templeton_aut_create_short_eval_data_crpo-sft-llama-3.1-8b-instruct_20260527_111159.json"
       "../outputs/templeton_aut/aut/crpo-dpo-llama-3.1-8b-instruct/20260527_111218/templeton_aut_create_short_eval_data_crpo-dpo-llama-3.1-8b-instruct_20260527_111218.json"
       
       ###################### network and random ablation experiments ######################
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260325_152659/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260325_152659.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260325_152802/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260325_152802.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260328_110605/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260328_110605.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260328_110713/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260328_110713.json"

       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260325_152820/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260325_152820.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260325_152941/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260325_152941.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260328_110731/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260328_110731.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260328_111309/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260328_111309.json"

       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260327_123950/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260327_123950.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260327_124553/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260327_124553.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260328_111931/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260328_111931.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260328_112526/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260328_112526.json"

       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260327_125200/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260327_125200.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260327_130046/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260327_130046.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260328_113126/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260328_113126.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260328_113957/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260328_113957.json"

       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260327_152841/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260327_152841.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260327_154020/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260327_154020.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260328_114815/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260328_114815.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260328_120003/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260328_120003.json"

       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260327_155142/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260327_155142.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260327_161118/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260327_161118.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260328_121105/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260328_121105.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260328_123003/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260328_123003.json"

       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260327_162909/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260327_162909.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260327_165816/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260327_165816.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260328_124724/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260328_124724.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260328_131628/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260328_131628.json"

       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260327_131633/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260327_131633.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260327_131653/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260327_131653.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260328_134139/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260328_134139.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260328_134155/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260328_134155.json"

       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260327_131707/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260327_131707.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260327_131742/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260327_131742.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260328_134211/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260328_134211.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260328_134240/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260328_134240.json"

       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260327_131757/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260327_131757.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260327_132247/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260327_132247.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260328_134255/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260328_134255.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260328_134648/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260328_134648.json"

       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260327_132338/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260327_132338.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260327_175537/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260327_175537.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260328_134742/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260328_134742.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260328_143414/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260328_143414.json"
)

for input_path in "${input_paths[@]}"; do
       python -m cadabra.model.compute_metrics -d "$input_path" -m "gemini_aut_score"
done