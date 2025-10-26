from mmengine.config import read_base

with read_base():
    # Models
    # Datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets
        
    from ...models.llama_31_8b import \
        llama_31_8b_flexprefill_models as llama3_1_8b_flexprefill_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = 128 * 1024
    model['run_cfg']['num_gpus'] = 1


work_dir = './results/longbenchv2/exps_llama_31_8b/flexprefill'