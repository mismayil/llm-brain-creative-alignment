#!/bin/bash

input_paths=(
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20251216_194701/templeton_aut_create_eval_data_llama-3.1-8b-instruct_20251216_194701.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260205_150428/templeton_aut_create_eval_data_with_subjects_llama-3.1-8b-instruct_20260205_150428.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260306_140725/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260306_140725.json"
       # "../outputs/templeton_aut/oct/llama-3.1-8b-instruct/20251216_195707/templeton_aut_object_eval_data_llama-3.1-8b-instruct_20251216_195707.json"
       # "../outputs/templeton_aut/oct/llama-3.1-8b-instruct/20260326_133627/templeton_aut_object_short_eval_data_llama-3.1-8b-instruct_20260326_133627.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260528_044935/templeton_aut_nolang_eval_data_llama-3.1-8b-instruct_20260528_044935.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260528_000048/templeton_aut_empty_eval_data_llama-3.1-8b-instruct_20260528_000048.json"
       
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20251216_202121/templeton_aut_create_eval_data_crpo-llama-3.1-8b-instruct-cre_20251216_202121.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260205_150442/templeton_aut_create_eval_data_with_subjects_crpo-llama-3.1-8b-instruct-cre_20260205_150442.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260306_140819/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260306_140819.json"
       # "../outputs/templeton_aut/oct/crpo-llama-3.1-8b-instruct-cre/20251216_203018/templeton_aut_object_eval_data_crpo-llama-3.1-8b-instruct-cre_20251216_203018.json"
       # "../outputs/templeton_aut/oct/crpo-llama-3.1-8b-instruct-cre/20260326_133808/templeton_aut_object_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260326_133808.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260528_045040/templeton_aut_nolang_eval_data_crpo-llama-3.1-8b-instruct-cre_20260528_045040.json"
       # "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260528_000200/templeton_aut_empty_eval_data_crpo-llama-3.1-8b-instruct-cre_20260528_000200.json"

       # # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260315_150050/templeton_aut_create_eval_data_gemma-3-270m-it_20260315_150050.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260316_153058/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260316_153058.json"
       # # "../outputs/templeton_aut/oct/gemma-3-270m-it/20260315_161236/templeton_aut_object_eval_data_gemma-3-270m-it_20260315_161236.json"
       # "../outputs/templeton_aut/oct/gemma-3-270m-it/20260316_163539/templeton_aut_object_short_eval_data_gemma-3-270m-it_20260316_163539.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260528_045114/templeton_aut_nolang_eval_data_gemma-3-270m-it_20260528_045114.json"
       # "../outputs/templeton_aut/aut/gemma-3-270m-it/20260528_000839/templeton_aut_empty_eval_data_gemma-3-270m-it_20260528_000839.json"
       
       # # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260315_150626/templeton_aut_create_eval_data_gemma-3-1b-it_20260315_150626.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260316_153639/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260316_153639.json"
       # # "../outputs/templeton_aut/oct/gemma-3-1b-it/20260315_161804/templeton_aut_object_eval_data_gemma-3-1b-it_20260315_161804.json"
       # "../outputs/templeton_aut/oct/gemma-3-1b-it/20260316_164110/templeton_aut_object_short_eval_data_gemma-3-1b-it_20260316_164110.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260528_045640/templeton_aut_nolang_eval_data_gemma-3-1b-it_20260528_045640.json"
       # "../outputs/templeton_aut/aut/gemma-3-1b-it/20260528_001415/templeton_aut_empty_eval_data_gemma-3-1b-it_20260528_001415.json"

       # # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260315_151407/templeton_aut_create_eval_data_gemma-3-4b-it_20260315_151407.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260316_154445/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260316_154445.json"
       # # "../outputs/templeton_aut/oct/gemma-3-4b-it/20260315_162545/templeton_aut_object_eval_data_gemma-3-4b-it_20260315_162545.json"
       # "../outputs/templeton_aut/oct/gemma-3-4b-it/20260316_164908/templeton_aut_object_short_eval_data_gemma-3-4b-it_20260316_164908.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260528_050419/templeton_aut_nolang_eval_data_gemma-3-4b-it_20260528_050419.json"
       # "../outputs/templeton_aut/aut/gemma-3-4b-it/20260528_002218/templeton_aut_empty_eval_data_gemma-3-4b-it_20260528_002218.json"

       # # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260315_152448/templeton_aut_create_eval_data_gemma-3-12b-it_20260315_152448.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260316_155546/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260316_155546.json"
       # # "../outputs/templeton_aut/oct/gemma-3-12b-it/20260315_163555/templeton_aut_object_eval_data_gemma-3-12b-it_20260315_163555.json"
       # "../outputs/templeton_aut/oct/gemma-3-12b-it/20260316_165927/templeton_aut_object_short_eval_data_gemma-3-12b-it_20260316_165927.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260528_051434/templeton_aut_nolang_eval_data_gemma-3-12b-it_20260528_051434.json"
       # "../outputs/templeton_aut/aut/gemma-3-12b-it/20260528_003239/templeton_aut_empty_eval_data_gemma-3-12b-it_20260528_003239.json"

       # # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260315_154131/templeton_aut_create_eval_data_gemma-3-27b-it_20260315_154131.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260316_161156/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260316_161156.json"
       # # "../outputs/templeton_aut/oct/gemma-3-27b-it/20260315_165046/templeton_aut_object_eval_data_gemma-3-27b-it_20260315_165046.json"
       # "../outputs/templeton_aut/oct/gemma-3-27b-it/20260316_171436/templeton_aut_object_short_eval_data_gemma-3-27b-it_20260316_171436.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260528_052816/templeton_aut_nolang_eval_data_gemma-3-27b-it_20260528_052816.json"
       # "../outputs/templeton_aut/aut/gemma-3-27b-it/20260528_004957/templeton_aut_empty_eval_data_gemma-3-27b-it_20260528_004957.json"

       # # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260315_160647/templeton_aut_create_eval_data_llama-3.2-1b-instruct_20260315_160647.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260316_163439/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260316_163439.json"
       # # "../outputs/templeton_aut/oct/llama-3.2-1b-instruct/20260315_171134/templeton_aut_object_eval_data_llama-3.2-1b-instruct_20260315_171134.json"
       # "../outputs/templeton_aut/oct/llama-3.2-1b-instruct/20260316_173525/templeton_aut_object_short_eval_data_llama-3.2-1b-instruct_20260316_173525.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260528_054628/templeton_aut_nolang_eval_data_llama-3.2-1b-instruct_20260528_054628.json"
       # "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260528_011302/templeton_aut_empty_eval_data_llama-3.2-1b-instruct_20260528_011302.json"

       # # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260315_160852/templeton_aut_create_eval_data_llama-3.2-3b-instruct_20260315_160852.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260316_163502/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260316_163502.json"
       # # "../outputs/templeton_aut/oct/llama-3.2-3b-instruct/20260315_171339/templeton_aut_object_eval_data_llama-3.2-3b-instruct_20260315_171339.json"
       # "../outputs/templeton_aut/oct/llama-3.2-3b-instruct/20260316_173558/templeton_aut_object_short_eval_data_llama-3.2-3b-instruct_20260316_173558.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260528_054826/templeton_aut_nolang_eval_data_llama-3.2-3b-instruct_20260528_054826.json"
       # "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260528_011329/templeton_aut_empty_eval_data_llama-3.2-3b-instruct_20260528_011329.json"

       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260322_134634/templeton_aut_create_eval_data_olmo-3.1-32b-instruct_20260322_134634.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260322_145141/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260322_145141.json"
       # "../outputs/templeton_aut/oct/olmo-3.1-32b-instruct/20260322_143229/templeton_aut_object_eval_data_olmo-3.1-32b-instruct_20260322_143229.json"
       # "../outputs/templeton_aut/oct/olmo-3.1-32b-instruct/20260322_145319/templeton_aut_object_short_eval_data_olmo-3.1-32b-instruct_20260322_145319.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260528_055223/templeton_aut_nolang_eval_data_olmo-3.1-32b-instruct_20260528_055223.json"
       # "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260528_011414/templeton_aut_empty_eval_data_olmo-3.1-32b-instruct_20260528_011414.json"

       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260322_141031/templeton_aut_create_eval_data_llama-3.1-70b-instruct_20260322_141031.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260322_145227/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260322_145227.json"
       # "../outputs/templeton_aut/oct/llama-3.1-70b-instruct/20260322_144137/templeton_aut_object_eval_data_llama-3.1-70b-instruct_20260322_144137.json"
       # "../outputs/templeton_aut/oct/llama-3.1-70b-instruct/20260322_145355/templeton_aut_object_short_eval_data_llama-3.1-70b-instruct_20260322_145355.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260528_055439/templeton_aut_nolang_eval_data_llama-3.1-70b-instruct_20260528_055439.json"
       # "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260528_012055/templeton_aut_empty_eval_data_llama-3.1-70b-instruct_20260528_012055.json"

       # "../outputs/templeton_aut/aut/mistral-7b-instruct-v0.3/20260325_175950/templeton_aut_create_eval_data_mistral-7b-instruct-v0.3_20260325_175950.json"
       # "../outputs/templeton_aut/aut/mistral-7b-instruct-v0.3/20260325_180953/templeton_aut_create_short_eval_data_mistral-7b-instruct-v0.3_20260325_180953.json"
       # "../outputs/templeton_aut/oct/mistral-7b-instruct-v0.3/20260325_180525/templeton_aut_object_eval_data_mistral-7b-instruct-v0.3_20260325_180525.json"
       # "../outputs/templeton_aut/oct/mistral-7b-instruct-v0.3/20260325_181042/templeton_aut_object_short_eval_data_mistral-7b-instruct-v0.3_20260325_181042.json"
       # "../outputs/templeton_aut/aut/mistral-7b-instruct-v0.3/20260802_004937/templeton_aut_empty_eval_data_mistral-7b-instruct-v0.3_20260802_004937.json"
       # "../outputs/templeton_aut/aut/mistral-7b-instruct-v0.3/20260802_005712/templeton_aut_nolang_eval_data_mistral-7b-instruct-v0.3_20260802_005712.json"

       # "../outputs/templeton_aut/aut/crpo-mistral-7b-instruct-cre/20260325_180411/templeton_aut_create_eval_data_crpo-mistral-7b-instruct-cre_20260325_180411.json"
       # "../outputs/templeton_aut/aut/crpo-mistral-7b-instruct-cre/20260325_181021/templeton_aut_create_short_eval_data_crpo-mistral-7b-instruct-cre_20260325_181021.json"
       # "../outputs/templeton_aut/oct/crpo-mistral-7b-instruct-cre/20260325_180925/templeton_aut_object_eval_data_crpo-mistral-7b-instruct-cre_20260325_180925.json"
       # "../outputs/templeton_aut/oct/crpo-mistral-7b-instruct-cre/20260325_181110/templeton_aut_object_short_eval_data_crpo-mistral-7b-instruct-cre_20260325_181110.json"

       # "../outputs/templeton_aut/aut/llama-3.1-8b/20260327_193626/templeton_aut_create_eval_data_llama-3.1-8b_20260327_193626.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b/20260327_200542/templeton_aut_create_short_eval_data_llama-3.1-8b_20260327_200542.json"
       # "../outputs/templeton_aut/oct/llama-3.1-8b/20260327_195036/templeton_aut_object_eval_data_llama-3.1-8b_20260327_195036.json"
       # "../outputs/templeton_aut/oct/llama-3.1-8b/20260327_201956/templeton_aut_object_short_eval_data_llama-3.1-8b_20260327_201956.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b/20260528_070157/templeton_aut_nolang_eval_data_llama-3.1-8b_20260528_070157.json"
       # "../outputs/templeton_aut/aut/llama-3.1-8b/20260528_031447/templeton_aut_empty_eval_data_llama-3.1-8b_20260528_031447.json"

       # "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260327_194332/templeton_aut_create_eval_data_llama-3.1-minitaur-8b_20260327_194332.json"
       # "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260327_201250/templeton_aut_create_short_eval_data_llama-3.1-minitaur-8b_20260327_201250.json"
       # "../outputs/templeton_aut/oct/llama-3.1-minitaur-8b/20260327_195754/templeton_aut_object_eval_data_llama-3.1-minitaur-8b_20260327_195754.json"
       # "../outputs/templeton_aut/oct/llama-3.1-minitaur-8b/20260327_202650/templeton_aut_object_short_eval_data_llama-3.1-minitaur-8b_20260327_202650.json"
       # "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260528_070722/templeton_aut_nolang_eval_data_llama-3.1-minitaur-8b_20260528_070722.json"
       # "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260528_032023/templeton_aut_empty_eval_data_llama-3.1-minitaur-8b_20260528_032023.json"

       # "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260328_174321/templeton_aut_create_eval_data_qwen2.5-32b-instruct_20260328_174321.json"
       # "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260328_210352/templeton_aut_create_short_eval_data_qwen2.5-32b-instruct_20260328_210352.json"
       # "../outputs/templeton_aut/oct/qwen2.5-32b-instruct/20260328_195820/templeton_aut_object_eval_data_qwen2.5-32b-instruct_20260328_195820.json"
       # # "../outputs/templeton_aut/oct/qwen2.5-32b-instruct/20260328_214659/templeton_aut_object_short_eval_data_qwen2.5-32b-instruct_20260328_214659.json"
       # "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260528_071244/templeton_aut_nolang_eval_data_qwen2.5-32b-instruct_20260528_071244.json"
       # "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260528_032603/templeton_aut_empty_eval_data_qwen2.5-32b-instruct_20260528_032603.json"

       # "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260328_175521/templeton_aut_create_eval_data_qwen2.5-72b-instruct_20260328_175521.json"
       # "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260328_210430/templeton_aut_create_short_eval_data_qwen2.5-72b-instruct_20260328_210430.json"
       # "../outputs/templeton_aut/oct/qwen2.5-72b-instruct/20260328_200408/templeton_aut_object_eval_data_qwen2.5-72b-instruct_20260328_200408.json"
       # # "../outputs/templeton_aut/oct/qwen2.5-72b-instruct/20260328_214729/templeton_aut_object_short_eval_data_qwen2.5-72b-instruct_20260328_214729.json"
       # "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260528_071456/templeton_aut_nolang_eval_data_qwen2.5-72b-instruct_20260528_071456.json"
       # "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260528_033234/templeton_aut_empty_eval_data_qwen2.5-72b-instruct_20260528_033234.json"

       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260328_184116/templeton_aut_create_eval_data_deepseek-r1-distill-llama-8b_20260328_184116.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260328_210527/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-8b_20260328_210527.json"
       # "../outputs/templeton_aut/oct/deepseek-r1-distill-llama-8b/20260328_201243/templeton_aut_object_eval_data_deepseek-r1-distill-llama-8b_20260328_201243.json"
       # "../outputs/templeton_aut/oct/deepseek-r1-distill-llama-8b/20260328_214818/templeton_aut_object_short_eval_data_deepseek-r1-distill-llama-8b_20260328_214818.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260528_043031/templeton_aut_empty_eval_data_deepseek-r1-distill-llama-8b_20260528_043031.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260528_131507/templeton_aut_nolang_eval_data_deepseek-r1-distill-llama-8b_20260528_131507.json"

       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260328_185026/templeton_aut_create_eval_data_deepseek-r1-distill-llama-70b_20260328_185026.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260328_211319/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-70b_20260328_211319.json"
       # "../outputs/templeton_aut/oct/deepseek-r1-distill-llama-70b/20260328_202039/templeton_aut_object_eval_data_deepseek-r1-distill-llama-70b_20260328_202039.json"
       # "../outputs/templeton_aut/oct/deepseek-r1-distill-llama-70b/20260328_215520/templeton_aut_object_short_eval_data_deepseek-r1-distill-llama-70b_20260328_215520.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260528_111903/templeton_aut_empty_eval_data_deepseek-r1-distill-llama-70b_20260528_111903.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260528_120041/templeton_aut_nolang_eval_data_deepseek-r1-distill-llama-70b_20260528_120041.json"

       # "../outputs/templeton_aut/aut/falcon-40b-instruct/20260328_193428/templeton_aut_create_eval_data_falcon-40b-instruct_20260328_193428.json"
       # "../outputs/templeton_aut/aut/falcon-40b-instruct/20260328_214322/templeton_aut_create_short_eval_data_falcon-40b-instruct_20260328_214322.json"
       # "../outputs/templeton_aut/oct/falcon-40b-instruct/20260328_205805/templeton_aut_object_eval_data_falcon-40b-instruct_20260328_205805.json"
       # # "../outputs/templeton_aut/oct/falcon-40b-instruct/20260328_222652/templeton_aut_object_short_eval_data_falcon-40b-instruct_20260328_222652.json"
       "../outputs/templeton_aut/aut/falcon-40b-instruct/20260802_095906/templeton_aut_empty_eval_data_falcon-40b-instruct_20260802_095906.json"
       # "../outputs/templeton_aut/aut/falcon-40b-instruct/20260528_123213/templeton_aut_nolang_eval_data_falcon-40b-instruct_20260528_123213.json"

       # "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260328_195016/templeton_aut_create_eval_data_qwen2.5-14b-instruct_20260328_195016.json"
       # "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260328_214636/templeton_aut_create_short_eval_data_qwen2.5-14b-instruct_20260328_214636.json"
       # "../outputs/templeton_aut/oct/qwen2.5-14b-instruct/20260328_205951/templeton_aut_object_eval_data_qwen2.5-14b-instruct_20260328_205951.json"
       # "../outputs/templeton_aut/oct/qwen2.5-14b-instruct/20260328_222850/templeton_aut_object_short_eval_data_qwen2.5-14b-instruct_20260328_222850.json"
       # "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260528_044624/templeton_aut_empty_eval_data_qwen2.5-14b-instruct_20260528_044624.json"
       # "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260528_132200/templeton_aut_nolang_eval_data_qwen2.5-14b-instruct_20260528_132200.json"

       # "../outputs/templeton_aut/aut/deepseek-r1-distill-qwen-7b/20260527_101959/templeton_aut_create_eval_data_deepseek-r1-distill-qwen-7b_20260527_101959.json"
       # "../outputs/templeton_aut/aut/deepseek-r1-distill-qwen-7b/20260527_105455/templeton_aut_create_short_eval_data_deepseek-r1-distill-qwen-7b_20260527_105455.json"

       # "../outputs/templeton_aut/aut/qwen2.5-math-7b/20260527_102718/templeton_aut_create_eval_data_qwen2.5-math-7b_20260527_102718.json"
       # "../outputs/templeton_aut/aut/qwen2.5-math-7b/20260527_110045/templeton_aut_create_short_eval_data_qwen2.5-math-7b_20260527_110045.json"

       # "../outputs/templeton_aut/aut/qwen2.5-7b/20260527_103459/templeton_aut_create_eval_data_qwen2.5-7b_20260527_103459.json"
       # "../outputs/templeton_aut/aut/qwen2.5-7b/20260527_110707/templeton_aut_create_short_eval_data_qwen2.5-7b_20260527_110707.json"

       # "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260527_104055/templeton_aut_create_eval_data_qwen2.5-7b-instruct_20260527_104055.json"
       # "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260527_111143/templeton_aut_create_short_eval_data_qwen2.5-7b-instruct_20260527_111143.json"
       # "../outputs/templeton_aut/oct/qwen2.5-7b-instruct/20260731_211145/templeton_aut_object_eval_data_qwen2.5-7b-instruct_20260731_211145.json"
       # "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260801_223234/templeton_aut_empty_eval_data_qwen2.5-7b-instruct_20260801_223234.json"
       # "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260801_224008/templeton_aut_nolang_eval_data_qwen2.5-7b-instruct_20260801_224008.json"

       # "../outputs/templeton_aut/aut/crpo-sft-llama-3.1-8b-instruct/20260527_104608/templeton_aut_create_eval_data_crpo-sft-llama-3.1-8b-instruct_20260527_104608.json"
       # "../outputs/templeton_aut/aut/crpo-sft-llama-3.1-8b-instruct/20260527_111159/templeton_aut_create_short_eval_data_crpo-sft-llama-3.1-8b-instruct_20260527_111159.json"

       # "../outputs/templeton_aut/aut/crpo-dpo-llama-3.1-8b-instruct/20260527_104800/templeton_aut_create_eval_data_crpo-dpo-llama-3.1-8b-instruct_20260527_104800.json"
       # "../outputs/templeton_aut/aut/crpo-dpo-llama-3.1-8b-instruct/20260527_111218/templeton_aut_create_short_eval_data_crpo-dpo-llama-3.1-8b-instruct_20260527_111218.json"

       # "../outputs/templeton_aut/aut/olmo-3-7b-instruct/20260801_093915/templeton_aut_create_eval_data_olmo-3-7b-instruct_20260801_093915.json"
       # "../outputs/templeton_aut/oct/olmo-3-7b-instruct/20260801_095146/templeton_aut_object_eval_data_olmo-3-7b-instruct_20260801_095146.json"
       # "../outputs/templeton_aut/aut/olmo-3-7b-instruct/20260801_100207/templeton_aut_create_short_eval_data_olmo-3-7b-instruct_20260801_100207.json"
       # "../outputs/templeton_aut/aut/olmo-3-7b-instruct/20260802_005549/templeton_aut_empty_eval_data_olmo-3-7b-instruct_20260802_005549.json"
       # "../outputs/templeton_aut/aut/olmo-3-7b-instruct/20260802_010039/templeton_aut_nolang_eval_data_olmo-3-7b-instruct_20260802_010039.json"

       # "../outputs/templeton_aut/aut/qwen2.5-0.5b-instruct/20260801_155606/templeton_aut_create_eval_data_qwen2.5-0.5b-instruct_20260801_155606.json"
       # "../outputs/templeton_aut/oct/qwen2.5-0.5b-instruct/20260801_160643/templeton_aut_object_eval_data_qwen2.5-0.5b-instruct_20260801_160643.json"
       # "../outputs/templeton_aut/aut/qwen2.5-0.5b-instruct/20260801_161344/templeton_aut_create_short_eval_data_qwen2.5-0.5b-instruct_20260801_161344.json"
       # "../outputs/templeton_aut/aut/qwen2.5-0.5b-instruct/20260801_223814/templeton_aut_empty_eval_data_qwen2.5-0.5b-instruct_20260801_223814.json"
       # "../outputs/templeton_aut/aut/qwen2.5-0.5b-instruct/20260801_224830/templeton_aut_nolang_eval_data_qwen2.5-0.5b-instruct_20260801_224830.json"

       # "../outputs/templeton_aut/aut/qwen2.5-1.5b-instruct/20260801_155831/templeton_aut_create_eval_data_qwen2.5-1.5b-instruct_20260801_155831.json"
       # "../outputs/templeton_aut/oct/qwen2.5-1.5b-instruct/20260801_160827/templeton_aut_object_eval_data_qwen2.5-1.5b-instruct_20260801_160827.json"
       # "../outputs/templeton_aut/aut/qwen2.5-1.5b-instruct/20260801_161435/templeton_aut_create_short_eval_data_qwen2.5-1.5b-instruct_20260801_161435.json"
       # "../outputs/templeton_aut/aut/qwen2.5-1.5b-instruct/20260801_223904/templeton_aut_empty_eval_data_qwen2.5-1.5b-instruct_20260801_223904.json"
       # "../outputs/templeton_aut/aut/qwen2.5-1.5b-instruct/20260801_224929/templeton_aut_nolang_eval_data_qwen2.5-1.5b-instruct_20260801_224929.json"

       # "../outputs/templeton_aut/aut/qwen2.5-3b-instruct/20260801_160201/templeton_aut_create_eval_data_qwen2.5-3b-instruct_20260801_160201.json"
       # "../outputs/templeton_aut/oct/qwen2.5-3b-instruct/20260801_161026/templeton_aut_object_eval_data_qwen2.5-3b-instruct_20260801_161026.json"
       # "../outputs/templeton_aut/aut/qwen2.5-3b-instruct/20260801_161453/templeton_aut_create_short_eval_data_qwen2.5-3b-instruct_20260801_161453.json"
       # "../outputs/templeton_aut/aut/qwen2.5-3b-instruct/20260801_223932/templeton_aut_empty_eval_data_qwen2.5-3b-instruct_20260801_223932.json"
       # "../outputs/templeton_aut/aut/qwen2.5-3b-instruct/20260801_225000/templeton_aut_nolang_eval_data_qwen2.5-3b-instruct_20260801_225000.json"
)

# full gen activation extraction
for input_path in "${input_paths[@]}"; do
       python -m cadabra.model.extract_lm_activations data_path="$input_path" batch_size=4
done

# prompt only activation extraction
for input_path in "${input_paths[@]}"; do
       python -m cadabra.model.extract_lm_activations data_path="$input_path" prompt_only=True
done