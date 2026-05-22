import torch
import torch.nn.functional as F
import math
import triton

from pbs_attn.src.kernels.permuted_block_sparse_attention import _permuted_block_sparse_attn_fwd, _permuted_block_sparse_attn_fwd_torch_naive
from pbs_attn.src.permute_states import apply_permutation
from pbs_attn.src.utils import block_pooled_attn, select_blocks

def first_token_mask(
    key_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Get the block index of the first token in the first segment.

    Args:
        key_indices: Tensor of shape (batch_size, num_heads, padded_kv_len)
        block_size: Size of the block
    
    Returns:
        Tensor of shape (batch_size, num_heads, 1, num_kv_blocks)
    """
    first_token_mask = (key_indices.view(key_indices.shape[0], key_indices.shape[1], -1, block_size) == 0).any(dim=-1)
    return first_token_mask[:, :, None, :]


_BLOCK_MASK_LAYOUT_CACHE = {}


def _get_block_mask_layout(
    q_block_num: int,
    kv_block_num: int,
    num_blocks_per_segment: int,
    causal: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (q_block_num, kv_block_num, num_blocks_per_segment, bool(causal), str(device))
    cached = _BLOCK_MASK_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached

    q_block_indices = torch.arange(q_block_num, device=device).unsqueeze(1)
    kv_block_indices = torch.arange(kv_block_num, device=device).unsqueeze(0)
    segment_mask = (q_block_indices // num_blocks_per_segment) == (
        kv_block_indices // num_blocks_per_segment
    )
    mask = ~segment_mask
    if causal:
        causal_mask = kv_block_indices <= q_block_indices + (kv_block_num - q_block_num)
        mask = mask & causal_mask

    _BLOCK_MASK_LAYOUT_CACHE[key] = (mask, segment_mask)
    return mask, segment_mask


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _gather_permuted_values(
    value_states: torch.Tensor,
    perm_key_indices: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    batch_size, num_kv_heads, kv_len, head_dim = value_states.shape
    _, num_q_heads, perm_len = perm_key_indices.shape

    if num_key_value_groups == 1:
        gather_indices = perm_key_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        return torch.gather(value_states, 2, gather_indices)

    kv_head_indices = (
        torch.arange(num_q_heads, device=perm_key_indices.device, dtype=perm_key_indices.dtype)
        // num_key_value_groups
    )
    flat_indices = perm_key_indices + (kv_head_indices * kv_len).view(1, num_q_heads, 1)
    flat_indices = flat_indices.reshape(batch_size, num_q_heads * perm_len)

    flat_values = value_states.contiguous().view(batch_size, num_kv_heads * kv_len, head_dim)
    flat_indices = flat_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    return torch.gather(flat_values, 1, flat_indices).view(batch_size, num_q_heads, perm_len, head_dim)


def permuted_block_selection(
    permuted_query_states: torch.Tensor,
    permuted_key_states: torch.Tensor,
    query_indices: torch.Tensor,
    key_indices: torch.Tensor,
    block_size: int,
    segment_size: int,
    threshold: float = 0.9,
    causal: bool = True,
    force_select_first_block: bool = True,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform permuted block selection using the given permuted query and key states.
    
    Args:
        permuted_query_states (torch.Tensor): Permuted query states of shape
                                              (batch_size, num_q_heads, q_len, head_dim).
        permuted_key_states (torch.Tensor): Permuted key states of shape
                                              (batch_size, num_kv_heads, kv_len, head_dim).
        query_indices (torch.Tensor): Query indices of shape (batch_size, num_q_heads, q_len).
        key_indices (torch.Tensor): Key indices of shape (batch_size, num_kv_heads, kv_len).
        block_size (int): Block size.
        segment_size (int): Segment size.
        threshold (float): Threshold for block selection.
        causal (bool): Whether to use causal attention.

    Returns:
        tuple: (block_attn_scores, block_mask, segment_mask)
    """
    # PBS: 2.1 Block Selection (Padding)
    batch_size, num_q_heads, q_len, head_dim = permuted_query_states.shape
    batch_size, num_kv_heads, kv_len, head_dim = permuted_key_states.shape

    assert num_q_heads == num_kv_heads
    assert q_len == kv_len, "Only support prefilling for now"
    assert segment_size % block_size == 0, "segment_size must be a multiple of block_size"
    q_num_to_pad = ((q_len + block_size - 1) // block_size) * block_size - q_len
    kv_num_to_pad = ((kv_len + block_size - 1) // block_size) * block_size - kv_len
    
    if q_num_to_pad > 0:
        padded_query_states = torch.nn.functional.pad(permuted_query_states, (0, 0, 0, q_num_to_pad), value=0)
    else:
        padded_query_states = permuted_query_states
        
    if kv_num_to_pad > 0:
        padded_key_states = torch.nn.functional.pad(permuted_key_states, (0, 0, 0, kv_num_to_pad), value=0)
        pad_indices = torch.arange(kv_len, kv_len + kv_num_to_pad, device=permuted_key_states.device)
        pad_indices = pad_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_kv_heads, -1)
        pad_key_indices = torch.cat([key_indices, pad_indices], dim=-1)
    else:
        padded_key_states = permuted_key_states
        pad_key_indices = key_indices
        
    padded_q_len = q_len + q_num_to_pad
    padded_kv_len = kv_len + kv_num_to_pad
        
    # PBS: 2.2 Block Selection (Mask Init)
    q_block_num = padded_q_len // block_size
    kv_block_num = padded_kv_len // block_size
    num_blocks_per_segment = segment_size // block_size

    mask, segment_mask = _get_block_mask_layout(
        q_block_num,
        kv_block_num,
        num_blocks_per_segment,
        causal,
        permuted_query_states.device,
    )

    # PBS: 2.3 Block Selection (Mean Pooling & Attention)
    block_attn_scores = block_pooled_attn(
        padded_query_states,
        padded_key_states,
        block_size,
        mask,
        query_pool_mode=query_pool_mode,
        key_pool_mode=key_pool_mode,
    )
    
    # PBS: 2.4 Block Selection (Select Blocks)
    if isinstance(threshold, int) and threshold == 1:
        block_mask = mask.view(1, 1, q_block_num, kv_block_num).expand_as(block_attn_scores)
    else:
        block_mask = select_blocks(block_attn_scores, threshold, causal)
        
    if force_select_first_block:
        # block_mask[:, :, :, 0] = True
        block_mask = block_mask | first_token_mask(pad_key_indices, block_size)
        

    return block_attn_scores, block_mask, segment_mask

def permuted_block_sparse_attn_fwd(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    segment_size: int,
    threshold: float,
    causal: bool,
    force_select_first_block: bool = True,
    use_triton: bool = True,
    # BLOCK_M: int = 64,
    # BLOCK_N: int = 64,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
    num_key_value_groups: int = 1,
):
    """
    Perform permuted block sparse attention forward pass.
    
    Args:
        query_states: Query tensor of shape (batch_size, num_q_heads, q_len, head_dim)
        key_states: Key tensor of shape (batch_size, num_kv_heads, kv_len, head_dim)
        value_states: Value tensor of shape (batch_size, num_kv_heads, kv_len, head_dim)
        block_size: Size of attention blocks
        segment_size: Size of segments for permutation
        threshold: Threshold for block selection
        causal: Whether to use causal attention
        force_select_first_block: Whether to force select first block
        use_triton: Whether to use Triton kernel implementation
        
        
    Returns:
        torch.Tensor: Attention output of shape (batch_size, num_q_heads, q_len, head_dim)
    """
    SEGMENT_SIZE = segment_size
    LOGICAL_BLOCK_SIZE = block_size

    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    batch_size, num_kv_heads, kv_len, head_dim = key_states.shape
    assert num_key_value_groups >= 1
    assert num_q_heads == num_kv_heads * num_key_value_groups
    assert causal
    assert q_len == kv_len

    # Fall back to regular attention if not permuting
    if q_len <= segment_size:
        if num_key_value_groups > 1:
            key_states = _repeat_kv(key_states, num_key_value_groups)
            value_states = _repeat_kv(value_states, num_key_value_groups)
        attn_outputs = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=causal,
        )
        return attn_outputs

    if num_key_value_groups > 1:
        key_states_expanded = _repeat_kv(key_states, num_key_value_groups)
    else:
        key_states_expanded = key_states
    
    # PBS: 1. Permutation Phase
    perm_key_states, perm_key_indices = apply_permutation(
        query_states=query_states,
        key_states=key_states_expanded,
        block_size=block_size,
        segment_size=segment_size,
    )
    # not permuting queries
    perm_query_states = query_states
    perm_query_indices = torch.arange(q_len, device=query_states.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_q_heads, -1)
    # PBS: 2. Block Selection
    block_attn_scores, block_mask, segment_mask = permuted_block_selection(
        permuted_query_states=perm_query_states,
        permuted_key_states=perm_key_states,
        query_indices=perm_query_indices,
        key_indices=perm_key_indices,
        block_size=block_size,
        segment_size=segment_size,
        threshold=threshold,
        causal=causal,
        force_select_first_block=force_select_first_block,
        query_pool_mode=query_pool_mode,
        key_pool_mode=key_pool_mode,
    )
    del block_attn_scores

    block_mask = block_mask | segment_mask[None, None, :, :]
    del segment_mask

    # PBS: 3. Attention Computation
    perm_attn_outputs = torch.empty_like(perm_query_states, device=perm_query_states.device)
    if use_triton:
        del perm_key_states

        def grid(META):
            return (triton.cdiv(q_len, META["BLOCK_M"]), num_q_heads, batch_size)

        _permuted_block_sparse_attn_fwd[grid](
            perm_query_states, key_states_expanded, value_states, perm_attn_outputs,
            block_mask,
            perm_query_indices, perm_key_indices,
            perm_query_states.stride(0), perm_query_states.stride(1), perm_query_states.stride(2), perm_query_states.stride(3),
            key_states_expanded.stride(0), key_states_expanded.stride(1), key_states_expanded.stride(2), key_states_expanded.stride(3),
            value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
            perm_attn_outputs.stride(0), perm_attn_outputs.stride(1), perm_attn_outputs.stride(2), perm_attn_outputs.stride(3),
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
            perm_query_indices.stride(0), perm_query_indices.stride(1), perm_query_indices.stride(2),
            perm_key_indices.stride(0), perm_key_indices.stride(1), perm_key_indices.stride(2),
            q_len, kv_len,
            1/math.sqrt(head_dim),
            num_q_heads,
            head_dim,
            LOGICAL_BLOCK_SIZE,
            SEGMENT_SIZE,
            num_key_value_groups,
            H=num_q_heads,
            num_kv_groups=num_key_value_groups,
            HEAD_DIM=head_dim,
            # BLOCK_M,
            # BLOCK_N,
            SEGMENT_SIZE=SEGMENT_SIZE,
            LOGICAL_BLOCK_SIZE=LOGICAL_BLOCK_SIZE,
            STAGE=3 if causal else 1,
        )
    else:
        perm_value_states = _gather_permuted_values(
            value_states,
            perm_key_indices,
            num_key_value_groups,
        )
        perm_value_indices = perm_key_indices
        perm_attn_outputs = _permuted_block_sparse_attn_fwd_torch_naive(
            perm_query_states, perm_key_states, perm_value_states, perm_attn_outputs,
            perm_query_indices, perm_key_indices, perm_value_indices,
            block_mask,
            block_size,
            segment_size,
            causal
        )

    return perm_attn_outputs
