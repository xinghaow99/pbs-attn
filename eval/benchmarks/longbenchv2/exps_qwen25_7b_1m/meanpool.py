from mmengine.config import read_base

with read_base():
    # Models
    # Datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets
        
    from ...models.qwen_25_7b_1m import \
        qwen2_5_7b_1m_meanpool_models as qwen2_5_7b_1m_meanpool_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = 400 * 1024 # truncated to avoid OOM
    model['model_kwargs']['tp_plan'] = 'auto'
    model['model_kwargs']['device_map'] = None
    model['run_cfg']['num_gpus'] = 4
    model['run_cfg']['num_procs'] = 4


work_dir = './results/longbenchv2/exps_qwen25_7b_1m/meanpool_400k'