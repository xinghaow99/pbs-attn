import torch
import torch.nn.functional as F
import math
from pbs_attn.src.utils import block_pooled_attn, select_blocks
from block_sparse_attn import block_sparse_attn_func

def MeanPooling_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int = 128,
    segment_size: int = 1024,
    threshold: float = 0.9,
    causal: bool = True,
    force_select_first_block: bool = True,
    force_select_current_block: bool = True,
):

    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    batch_size, num_kv_heads, kv_len, head_dim = key_states.shape
    assert num_q_heads == num_kv_heads
    assert q_len == kv_len, "Only support prefilling for now, q_len: {}, kv_len: {}".format(q_len, kv_len)
    assert batch_size == 1, "Only support batch size 1 for now"
    q_num_to_pad = ((q_len + block_size - 1) // block_size) * block_size - q_len
    kv_num_to_pad = ((kv_len + block_size - 1) // block_size) * block_size - kv_len

    if q_num_to_pad > 0:
        padded_query_states = torch.nn.functional.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)
    else:
        padded_query_states = query_states
    
    if kv_num_to_pad > 0:
        padded_key_states = torch.nn.functional.pad(key_states, (0, 0, 0, kv_num_to_pad), value=0)
    else:
        padded_key_states = key_states
    
    padded_q_len = q_len + q_num_to_pad
    padded_kv_len = kv_len + kv_num_to_pad
    
    q_block_num = padded_q_len // block_size
    kv_block_num = padded_kv_len // block_size
    
    mask = torch.ones(q_block_num, kv_block_num, device=query_states.device, dtype=torch.bool)
    if causal:
        causal_mask = torch.tril(torch.ones(q_block_num, kv_block_num, device=query_states.device, dtype=torch.bool), diagonal=kv_block_num - q_block_num)
        mask &= causal_mask
    
    block_attn_scores = block_pooled_attn(padded_query_states, padded_key_states, block_size, mask)
    
    block_mask = select_blocks(block_attn_scores, threshold, causal)

    if force_select_first_block:
        block_mask[:, :, :, 0] = True
    if force_select_current_block:

        cur_mask = torch.eye(q_block_num, kv_block_num, device=query_states.device, dtype=torch.bool)
        cur_mask = cur_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_q_heads, q_block_num, kv_block_num)
        block_mask |= cur_mask
    
    query_states = query_states.transpose(1, 2).view(q_len, num_q_heads, head_dim)
    key_states = key_states.transpose(1, 2).view(kv_len, num_kv_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(kv_len, num_kv_heads, head_dim)
    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, kv_len], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(num_q_heads)], device=query_states.device, dtype=torch.int32
    )
    assert block_mask.shape == (1, num_q_heads, q_block_num, kv_block_num), f"block_mask.shape: {block_mask.shape}, q_block_num: {q_block_num}, kv_block_num: {kv_block_num}"
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert block_mask.device == query_states.device

    
    attn_outputs_cuda = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        block_mask.contiguous(),
        q_len,
        kv_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    ).unsqueeze(0)

    attn_outputs = attn_outputs_cuda.reshape(batch_size, q_len, num_q_heads, head_dim).transpose(1, 2)

    return attn_outputs
