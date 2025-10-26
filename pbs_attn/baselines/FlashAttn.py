import torch
from flash_attn import flash_attn_func

def FlashAttn_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
    softmax_scale: float = None,
):
    q_transposed = query_states.transpose(1, 2)
    k_transposed = key_states.transpose(1, 2)
    v_transposed = value_states.transpose(1, 2)
    
    attn_output = flash_attn_func(
        q_transposed, 
        k_transposed, 
        v_transposed, 
        causal=causal,
        softmax_scale=softmax_scale
    )
    attn_output = attn_output.transpose(1, 2)
    
    return attn_output
