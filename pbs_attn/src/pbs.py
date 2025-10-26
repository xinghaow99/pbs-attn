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
        pad_indices = torch.arange(q_len, q_len + q_num_to_pad, device=permuted_query_states.device)
        pad_indices = pad_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_q_heads, -1)
        pad_query_indices = torch.cat([query_indices, pad_indices], dim=-1)

    else:
        padded_query_states = permuted_query_states
        pad_query_indices = query_indices
        pad_indices = None
        
    if kv_num_to_pad > 0:
        padded_key_states = torch.nn.functional.pad(permuted_key_states, (0, 0, 0, kv_num_to_pad), value=0)
        pad_indices = torch.arange(kv_len, kv_len + kv_num_to_pad, device=permuted_key_states.device)
        pad_indices = pad_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_kv_heads, -1)
        pad_key_indices = torch.cat([key_indices, pad_indices], dim=-1)
    else:
        padded_key_states = permuted_key_states
        pad_key_indices = key_indices
        pad_indices = None
        
    padded_q_len = q_len + q_num_to_pad
    padded_kv_len = kv_len + kv_num_to_pad
        
    # PBS: 2.2 Block Selection (Mask Init)
    q_block_num = padded_q_len // block_size
    kv_block_num = padded_kv_len // block_size
    num_blocks_per_segment = segment_size // block_size

    mask = torch.ones(q_block_num, kv_block_num, device=permuted_query_states.device, dtype=torch.bool)
    q_block_indices = torch.arange(q_block_num, device=permuted_query_states.device).unsqueeze(1)
    kv_block_indices = torch.arange(kv_block_num, device=permuted_key_states.device).unsqueeze(0)
    segment_mask = (q_block_indices // num_blocks_per_segment) == (kv_block_indices // num_blocks_per_segment)

    mask &= ~segment_mask
    if causal:
        causal_mask = kv_block_indices <= q_block_indices + (kv_block_num - q_block_num)
        mask &= causal_mask

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
        block_mask |= first_token_mask(pad_key_indices, block_size)
        

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
    assert num_q_heads == num_kv_heads
    assert causal
    

    # PBS: 1. Permutation Phase
    perm_key_states, perm_key_indices = apply_permutation(
        query_states=query_states,
        key_states=key_states,
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

    block_mask = block_mask | segment_mask[None, None, :, :]

    # PBS: 3. Value Permutation
    # Permute value states using the same indices as key states to keep them aligned
    indices_for_gather = perm_key_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    perm_value_states = torch.gather(value_states, 2, indices_for_gather)
    perm_value_indices = perm_key_indices

    # PBS: 4. Attention Computation
    num_kv_groups = num_q_heads // num_kv_heads
    perm_attn_outputs = torch.empty_like(perm_query_states, device=perm_query_states.device)
    if use_triton:
        def grid(META):
            return (triton.cdiv(q_len, META["BLOCK_M"]), num_q_heads, batch_size)

        _permuted_block_sparse_attn_fwd[grid](
            perm_query_states, perm_key_states, perm_value_states, perm_attn_outputs,
            block_mask,
            perm_query_indices, perm_key_indices, perm_value_indices,
            perm_query_states.stride(0), perm_query_states.stride(1), perm_query_states.stride(2), perm_query_states.stride(3),
            perm_key_states.stride(0), perm_key_states.stride(1), perm_key_states.stride(2), perm_key_states.stride(3),
            perm_value_states.stride(0), perm_value_states.stride(1), perm_value_states.stride(2), perm_value_states.stride(3),
            perm_attn_outputs.stride(0), perm_attn_outputs.stride(1), perm_attn_outputs.stride(2), perm_attn_outputs.stride(3),
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
            perm_query_indices.stride(0), perm_query_indices.stride(1), perm_query_indices.stride(2),
            perm_key_indices.stride(0), perm_key_indices.stride(1), perm_key_indices.stride(2),
            perm_value_indices.stride(0), perm_value_indices.stride(1), perm_value_indices.stride(2),
            q_len, kv_len,
            1/math.sqrt(head_dim),
            H=num_q_heads,
            num_kv_groups=num_kv_groups,
            HEAD_DIM=head_dim,
            # BLOCK_M,
            # BLOCK_N,
            SEGMENT_SIZE=SEGMENT_SIZE,
            LOGICAL_BLOCK_SIZE=LOGICAL_BLOCK_SIZE,
            STAGE=3 if causal else 1,
        )
    else:
        perm_attn_outputs = _permuted_block_sparse_attn_fwd_torch_naive(
            perm_query_states, perm_key_states, perm_value_states, perm_attn_outputs,
            perm_query_indices, perm_key_indices, perm_value_indices,
            block_mask,
            block_size,
            segment_size,
            causal
        )

    return perm_attn_outputs
