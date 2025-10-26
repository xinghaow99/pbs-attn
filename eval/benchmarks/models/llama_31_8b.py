import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.dirname(current_dir)
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

import opencompass_models
from opencompass_models import PatchedHuggingFaceCausalLM

os.environ['TOKENIZER_MODEL'] = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

llama_31_8b_flashattn_models = [
        dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-flashattn',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        patch_type='flashattn',
        patch_kwargs=dict(
            causal=True
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_minference_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-minference',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        patch_type='minference',
        patch_kwargs=dict(
            vertical_size=1000,
            slash_size=6096,
            adaptive_budget=None
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_flexprefill_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-flexprefill',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        patch_type='flexprefill',
        patch_kwargs=dict(
            gamma=0.95,
            tau=0.1,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_xattn_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-xattn',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        patch_type='xattention',
        patch_kwargs=dict(
            stride=8,
            threshold=0.9,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_meanpooling_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-meanpool',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        patch_type='meanpooling',
        patch_kwargs=dict(
            block_size=128,
            threshold=0.9,
            force_select_first_block=True,
            force_select_current_block=True
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    )
]

llama_31_8b_pbs_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-pbs',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        patch_type='pbs',
        patch_kwargs=dict(
            block_size=128,
            segment_size=256,
            threshold=0.9,
            force_select_first_block=True,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]