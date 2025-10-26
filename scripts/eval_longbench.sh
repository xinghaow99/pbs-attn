cd eval/benchmarks

########################################################
# Llama 3.1 8B Instruct
########################################################
# PBS-Attn
opencompass longbench/exps_llama_31_8b/pbs.py --max-num-workers 8

# Flash Attention
opencompass longbench/exps_llama_31_8b/flashattn.py --max-num-workers 8
# Minference
opencompass longbench/exps_llama_31_8b/minference.py --max-num-workers 8
# FlexPrefill
opencompass longbench/exps_llama_31_8b/flexprefill.py --max-num-workers 8
# XAttention
opencompass longbench/exps_llama_31_8b/xattn.py --max-num-workers 8
# MeanPooling
opencompass longbench/exps_llama_31_8b/meanpooling.py --max-num-workers 8

########################################################
# Qwen 2.5 7B 1M
########################################################
# PBS-Attn
opencompass longbench/exps_qwen_25_7b_1m/pbs.py --max-num-workers 8

# Flash Attention
opencompass longbench/exps_qwen_25_7b_1m/flashattn.py --max-num-workers 8
# Minference
opencompass longbench/exps_qwen_25_7b_1m/minference.py --max-num-workers 8
# FlexPrefill
opencompass longbench/exps_qwen_25_7b_1m/flexprefill.py --max-num-workers 8
# XAttention
opencompass longbench/exps_qwen_25_7b_1m/xattn.py --max-num-workers 8
# MeanPooling
opencompass longbench/exps_qwen_25_7b_1m/meanpooling.py --max-num-workers 8