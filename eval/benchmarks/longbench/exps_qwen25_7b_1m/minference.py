from mmengine.config import read_base

with read_base():
    # Models
    # Datasets
    from opencompass.configs.datasets.longbench.longbench import \
        longbench_datasets as LongBench_datasets
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups
    from ...models.qwen_25_7b_1m import \
        qwen2_5_7b_1m_minference_models as qwen2_5_7b_1m_minference_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = 1024 * 1024
    model['run_cfg']['num_gpus'] = 1

summarizer = dict(
    dataset_abbrs=[
        ['longbench', 'naive_average'],
        ['longbench_zh', 'naive_average'],
        ['longbench_en', 'naive_average'],
        '',
        'longbench_single-document-qa',
        'longbench_multi-document-qa',
        'longbench_summarization',
        'longbench_few-shot-learning',
        'longbench_synthetic-tasks',
        'longbench_code-completion',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

work_dir = './results/longbench/exps_qwen25_7b_1m/minference'