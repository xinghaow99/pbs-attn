import torch
import torch.nn.functional as F

import math

def last_block_attn_sorting(
    queries: torch.Tensor,
    keys: torch.Tensor,
    block_size: int,
    segment_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort keys per segment by attention to the last query block.

    Args:
        queries: (B, H, S, D) queries.
        keys: (B, H, S, D) keys to permute.
        block_size: Block size for Flash Attention.
        segment_size: Segment size for permutation.

    Returns:
        (permuted_keys, indices) where permuted_keys is (B, H, S, D) and
        indices is (B, H, S).
    """

    batch_size, num_heads, seq_len, head_dim = keys.shape
    device, dtype = keys.device, keys.dtype
    
    assert segment_size > 0 and block_size > 0
    
    num_complete_segments = seq_len // segment_size
    remainder_size = seq_len % segment_size
    complete_seq_len = num_complete_segments * segment_size
    
    # No-op cases: nothing to sort
    if num_complete_segments == 0 or (num_complete_segments == 1 and remainder_size == 0):
        identity_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        return keys, identity_indices

    states_complete = keys[:, :, :complete_seq_len, :]
    states_remainder = keys[:, :, complete_seq_len:, :] if remainder_size > 0 else None
    
    # Prepare Queries
    last_block_queries = queries[:, :, -block_size:, :]

    # Compute Scores & Sort
    keys_complete = keys[:, :, :complete_seq_len, :]
    attn_scores = torch.matmul(last_block_queries, keys_complete.transpose(-1, -2)) / math.sqrt(head_dim)
    attn_probs_complete = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    avg_key_scores_complete = attn_probs_complete.mean(dim=-2)
    
    scores_reshaped = avg_key_scores_complete.reshape(
        batch_size, avg_key_scores_complete.shape[1], num_complete_segments, segment_size
    )
    
    sorted_indices = torch.argsort(scores_reshaped, dim=-1, descending=True)

    # Apply Permutation
    states_5d = states_complete.reshape(batch_size, num_heads, num_complete_segments, segment_size, head_dim)
    indices_for_gather = sorted_indices.unsqueeze(-1).expand_as(states_5d)
    sorted_states_5d = torch.gather(states_5d, dim=3, index=indices_for_gather)
    sorted_complete = sorted_states_5d.reshape(batch_size, num_heads, complete_seq_len, head_dim)

    # Final Assembly
    final_result = (
        torch.cat([sorted_complete, states_remainder], dim=2)
        if remainder_size > 0
        else sorted_complete
    )

    segment_start_offsets = torch.arange(0, complete_seq_len, segment_size, device=device).view(1, 1, num_complete_segments, 1)
    global_indices_complete = (sorted_indices + segment_start_offsets).reshape(batch_size, num_heads, complete_seq_len)

    final_indices_to_return = (
        torch.cat([
            global_indices_complete,
            torch.arange(complete_seq_len, seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        ], dim=2)
        if remainder_size > 0
        else global_indices_complete
    )
    
    return final_result, final_indices_to_return



def apply_permutation(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    segment_size: int,
    
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Permute keys using last_block_attn_sorting with the given queries.

    Returns (permuted_keys, indices).
    """
    permuted_key_states, key_indices = last_block_attn_sorting(
        queries=query_states,
        keys=key_states,
        block_size=block_size,
        segment_size=segment_size,
    )

    return permuted_key_states, key_indices
