cd eval/benchmarks

########################################################
# Llama 3.1 8B Instruct
########################################################
# PBS-Attn
opencompass longbenchv2/exps_llama_31_8b/pbs.py --max-num-workers 8

# Flash Attention
opencompass longbenchv2/exps_llama_31_8b/flashattn.py --max-num-workers 8
# Minference
opencompass longbenchv2/exps_llama_31_8b/minference.py --max-num-workers 8
# FlexPrefill
opencompass longbenchv2/exps_llama_31_8b/flexprefill.py --max-num-workers 8
# XAttention
opencompass longbenchv2/exps_llama_31_8b/xattn.py --max-num-workers 8
# MeanPooling
opencompass longbenchv2/exps_llama_31_8b/meanpooling.py --max-num-workers 8

########################################################
# Qwen 2.5 7B 1M (TP=4 since there are only 4 kv heads)
########################################################
# PBS-Attn
opencompass longbenchv2/exps_qwen_25_7b_1m/pbs.py --max-num-workers 4

# Flash Attention
opencompass longbenchv2/exps_qwen_25_7b_1m/flashattn.py --max-num-workers 4
# Minference
opencompass longbenchv2/exps_qwen_25_7b_1m/minference.py --max-num-workers 4
# FlexPrefill
opencompass longbenchv2/exps_qwen_25_7b_1m/flexprefill.py --max-num-workers 4
# XAttention
opencompass longbenchv2/exps_qwen_25_7b_1m/xattn.py --max-num-workers 4
# MeanPooling
opencompass longbenchv2 /exps_qwen_25_7b_1m/meanpooling.py --max-num-workers 4