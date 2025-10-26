import torch
import torch.nn.functional as F
import math
from typing import Union

def block_pooled_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    mask: torch.Tensor,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
):
    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, kv_len, _ = key_states.shape
    q_block_num = q_len // block_size
    kv_block_num = kv_len // block_size
    if query_pool_mode == "mean":
        pooled_query_states = query_states.reshape(batch_size, num_q_heads, q_block_num, block_size, head_dim).mean(dim=-2)
    elif query_pool_mode == "max":
        pooled_query_states = query_states.reshape(batch_size, num_q_heads, q_block_num, block_size, head_dim).max(dim=-2).values
    else:
        raise ValueError(f"Invalid query_pool_mode: {query_pool_mode}")
    if key_pool_mode == "mean":
        pooled_key_states = key_states.reshape(batch_size, num_kv_heads, kv_block_num, block_size, head_dim).mean(dim=-2)
    elif key_pool_mode == "max":
        pooled_key_states = key_states.reshape(batch_size, num_kv_heads, kv_block_num, block_size, head_dim).max(dim=-2).values
    else:
        raise ValueError(f"Invalid key_pool_mode: {key_pool_mode}")
    block_attn_scores = torch.einsum("bhqd,bhkd->bhqk", pooled_query_states, pooled_key_states)
    block_attn_scores /= math.sqrt(head_dim)
    block_attn_scores = block_attn_scores.masked_fill(~mask, float("-inf"))
    block_attn_scores = F.softmax(block_attn_scores, dim=-1, dtype=torch.float32)
    return block_attn_scores



def select_blocks(
    block_attn_scores: torch.Tensor,
    threshold: Union[float, torch.Tensor],
    causal: bool = True,
) -> torch.Tensor:
    """
    Select the blocks to attend to based on cumulative attention scores.

    Args:
        block_attn_scores: (batch_size, num_heads, q_block_num, kv_block_num)
        threshold: float, the threshold for cumulative attention scores.
        causal: bool, Must be True. This implementation is only for causal selection.

    Returns:
        block_mask: (batch_size, num_heads, q_block_num, kv_block_num)
    """
    assert causal == True, "This implementation variant strictly supports causal=True."

    batch_size, num_heads, q_block_num, kv_block_num = block_attn_scores.shape
    device = block_attn_scores.device
    # fill nans to zeros
    block_attn_scores = torch.nan_to_num(block_attn_scores, nan=0.0)
    if q_block_num == 0 or kv_block_num == 0:
        return torch.zeros_like(block_attn_scores, dtype=torch.bool, device=device)

    # Step 1: Sort scores and get original indices
    sorted_scores, sorted_indices = torch.sort(block_attn_scores, dim=-1, descending=True)

    # Step 2: Calculate cumulative scores
    cumulative_scores = torch.cumsum(sorted_scores, dim=-1)

    # Step 3: Identify blocks meeting the threshold
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
    exceed_threshold = cumulative_scores >= threshold
    # Step 4: Find the first index (sorted order) where cumulative score exceeds threshold
    indices_first_exceed = torch.argmax(exceed_threshold.int(), dim=-1, keepdim=True)
    # Step 5: Check if threshold was actually met
    any_exceeds = torch.any(exceed_threshold, dim=-1, keepdim=True)

    # Step 6: Create selection mask in sorted order
    ramp = torch.arange(kv_block_num, device=device).view(1, 1, 1, kv_block_num)
    selected_mask_sorted = (ramp <= indices_first_exceed) & any_exceeds

    # Step 7: Scatter selection mask back to original block positions
    output_block_mask = torch.empty_like(block_attn_scores, dtype=torch.bool, device=device)
    output_block_mask.scatter_(dim=-1, index=sorted_indices, src=selected_mask_sorted)

    return output_block_mask
