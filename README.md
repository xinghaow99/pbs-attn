# Sparser Block-Sparse Attention via Token Permutation

## ✨ TL;DR
We introduce Permuted Block-Sparse Attention (PBS-Attn), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling.

## Core Concepts
We apply token permutation which allows us to select far fewer blocks while still covering the important key tokens, which look like the following:
![attn_map](./assets/attn_map.png)

The permutation is done within segments so that we don't break causality for LLMs:
![pbs_attn](./assets/pbs_attn.png)


## 🚀 Setup
### Main Installation
Get started with a one-liner installation of PBS-Attn:
```
pip install -e .
```
### Dependencies
To reproduce the results in the paper, you'll need OpenCompass for evaluation:
```
pip install datasets==3.6.0 opencompass==0.4.2
```

To run the baseline comparisons, please install their respective packages:
```
# Flash Attention
pip install ninja
pip install flash-attn --no-build-isolation

# Minference
pip install minference

# XAttention 
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
python setup.py install
cd ..
```

## 📊 Evaluation
We provide one-liner scripts for easy reproduction. You can also modify the scripts for specific evaluations (e.g., certain baselines, different setups).
### LongBench
```
bash scripts/eval_longbench.sh
```
### LongBenchv2
```
bash scripts/eval_longbenchv2.sh
```
### Efficiency
```
bash scripts/eval_efficiency.sh
```
## 📜 Citation
If you find our code useful, please kindly cite our paper as the following:
```
[TODO]
```