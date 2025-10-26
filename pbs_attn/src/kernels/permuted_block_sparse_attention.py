import torch
import triton
import triton.language as tl
import math


@triton.jit
def _permuted_block_sparse_attn_fwd_inner(
    acc, l_i, m_i,
    q,
    qo_len,
    kv_len,
    K_ptrs,
    V_ptrs,
    K_block_indices_ptrs,
    perm_Q_indices_ptrs,
    perm_K_indices_ptrs,
    perm_V_indices_ptrs,
    stride_seq_k,
    stride_seq_v,
    stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
    stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
    stride_bz_perm_v_indices, stride_h_perm_v_indices, stride_seq_perm_v_indices,
    pid_seq,
    RANGE_Q_SEQ,
    RANGE_KV_SEQ,
    softmax_scale,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
):
    segment_id =(pid_seq*BLOCK_M) // SEGMENT_SIZE
    if STAGE == 1:
        lo, hi = 0, (segment_id * SEGMENT_SIZE)
        # perm_Q_indices = None
    elif STAGE == 2:
        lo = segment_id * SEGMENT_SIZE
        hi = tl.minimum(lo + SEGMENT_SIZE, kv_len)

        perm_Q_indices = tl.load(perm_Q_indices_ptrs, boundary_check=(0,1))
    elif STAGE == 3:
        lo, hi = 0, kv_len
        # perm_Q_indices = None

    K_ptrs_cur = tl.advance(K_ptrs, (0, lo))
    V_ptrs_cur = tl.advance(V_ptrs, (lo, 0))
    perm_K_indices_ptrs_cur = tl.advance(perm_K_indices_ptrs, (0, lo))
    

    if not (STAGE == 1 and segment_id == 0):
        for kv_seq_start in range(lo, hi, BLOCK_N):
            k_block_idx = tl.load(K_block_indices_ptrs + kv_seq_start // LOGICAL_BLOCK_SIZE)
            if k_block_idx:

                kv_mask = RANGE_KV_SEQ[None, :] >= (kv_len - kv_seq_start) # True if exceed kv_len 
                k = tl.load(K_ptrs_cur, boundary_check=(0,1), padding_option="zero") # (BLOCK_N, HEAD_DIM)
                qk = tl.dot(q, k) 
                qk *= softmax_scale
                if STAGE == 2:
                    # On-diagonal segments, apply mask within segment
                    perm_K_indices = tl.load(perm_K_indices_ptrs_cur, boundary_check=(0,1), padding_option="zero")
                    mask = perm_Q_indices < perm_K_indices # (BLOCK_M, BLOCK_N)
                    kv_mask |= mask

                qk = qk + tl.where(kv_mask, -1e6, 0)
                local_m = tl.max(qk, 1)
                m_ij = tl.maximum(m_i, local_m)
                qk -= m_ij[:, None]
                
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(m_i-m_ij)

                acc = acc * alpha[:, None]
                v = tl.load(V_ptrs_cur, boundary_check=(0,1), padding_option="zero")
                p = p.to(dtype)

                acc += tl.dot(p, v)
                l_i = l_i * alpha + l_ij
                m_i = m_ij
            
            K_ptrs_cur = tl.advance(K_ptrs_cur, (0, BLOCK_N))
            V_ptrs_cur = tl.advance(V_ptrs_cur, (BLOCK_N, 0))
            perm_K_indices_ptrs_cur = tl.advance(perm_K_indices_ptrs_cur, (0, BLOCK_N))
    
    return acc, l_i, m_i


def _prune_invalid_configs_permuted(configs, named_args, **kwargs):
    # Triton passes autotune key args via kwargs; fall back to named_args if needed
    logical_bs = kwargs.get('LOGICAL_BLOCK_SIZE', None)
    if logical_bs is None:
        logical_bs = named_args.get('LOGICAL_BLOCK_SIZE', None)
    try:
        logical_bs = int(logical_bs)
    except Exception:
        logical_bs = None
    if logical_bs is None or logical_bs <= 0:
        return configs
    pruned = []
    for conf in configs:
        bm = conf.kwargs.get('BLOCK_M', 0)
        bn = conf.kwargs.get('BLOCK_N', 0)
        if bm == 0 or bn == 0:
            continue
        # Require both tiles to divide logical block size to avoid crossing logical-block boundaries
        if (logical_bs % bm == 0) and (logical_bs % bn == 0):
            pruned.append(conf)
    return pruned


@triton.autotune(
    configs=[
        # Basic configurations
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),

        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
        # Configurations with more stages for potential latency hiding
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 3}),
   ],
    key=['H', 'HEAD_DIM', 'LOGICAL_BLOCK_SIZE', 'SEGMENT_SIZE', 'num_kv_groups'],
    prune_configs_by={'early_config_prune': _prune_invalid_configs_permuted},
)
@triton.jit
def _permuted_block_sparse_attn_fwd(
    perm_Q, perm_K, perm_V, perm_O, # (batch_size, num_q_heads(num_kv_heads), q_len(kv_len), head_dim)
    K_block_indices, # (batch_size, num_q_heads, num_q_blocks, num_k_blocks)
    perm_Q_indices, # (batch_size, num_q_heads, q_len)
    perm_K_indices, # (batch_size, num_kv_heads, kv_len)
    perm_V_indices, # (batch_size, num_kv_heads, kv_len)
    stride_bz_q, stride_h_q, stride_seq_q, stride_d_q,
    stride_bz_k, stride_h_k, stride_seq_k, stride_d_k,
    stride_bz_v, stride_h_v, stride_seq_v, stride_d_v,
    stride_bz_o, stride_h_o, stride_seq_o, stride_d_o,
    stride_bz_k_block_indices, stride_h_k_block_indices, stride_seqq_k_block_indices, stride_seqk_k_block_indices,
    stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
    stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
    stride_bz_perm_v_indices, stride_h_perm_v_indices, stride_seq_perm_v_indices,
    qo_len, kv_len,
    softmax_scale,
    H:tl.constexpr,
    num_kv_groups:tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  
    BLOCK_N: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr
):
    
    # Enforce tile alignment with logical block size at compile-time
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_M) == 0)
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_N) == 0)
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)


    range_Q_seq = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    range_KV_seq = tl.arange(0, BLOCK_N)
    range_h = tl.arange(0, HEAD_DIM)

    dtype = perm_Q.type.element_ty

    # Init ptrs
    Q_ptrs = tl.make_block_ptr(
        base=perm_Q+pid_bz*stride_bz_q+pid_h*stride_h_q,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_seq_q, stride_d_q),
        offsets=(pid_seq*BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_ptrs = tl.make_block_ptr(
        base=perm_K + pid_bz * stride_bz_k + (pid_h // num_kv_groups) * stride_h_k,
        shape=(HEAD_DIM, kv_len),
        strides=(stride_d_k, stride_seq_k),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0)
    )
    V_ptrs = tl.make_block_ptr(
        base=perm_V + pid_bz * stride_bz_v + (pid_h // num_kv_groups) * stride_h_v,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_seq_v, stride_d_v),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    O_ptrs = tl.make_block_ptr(
        base=perm_O + pid_bz * stride_bz_o + pid_h * stride_h_o,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_seq_o, stride_d_o),
        offsets=(pid_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    # Calculate the starting query token index for this program
    q_start_index = pid_seq * BLOCK_M
    # Calculate the logical query block this program belongs to.
    # Because we asserted LOGICAL_BLOCK_SIZE % BLOCK_M == 0, this is guaranteed to be a single value.
    logical_q_block_idx = q_start_index // LOGICAL_BLOCK_SIZE
    K_block_indices_ptrs = K_block_indices + pid_bz*stride_bz_k_block_indices + pid_h*stride_h_k_block_indices + logical_q_block_idx*stride_seqq_k_block_indices
    # K_block_indices_ptrs = K_block_indices + (pid_bz * (H // num_kv_groups) + pid_h // num_kv_groups) * stride_h_k_block_indices + pid_seq*stride_seqq_k_block_indices
    perm_Q_indices_ptrs = tl.make_block_ptr(
        base=perm_Q_indices + pid_bz*stride_bz_perm_q_indices + pid_h*stride_h_perm_q_indices,
        shape=(qo_len, tl.constexpr(1)),
        strides=(stride_seq_perm_q_indices, tl.constexpr(0)),
        offsets=(pid_seq*BLOCK_M, 0),
        block_shape=(BLOCK_M, tl.constexpr(1)),
        order=(1, 0)
    ) # (BLOCK_M, 1)
    perm_K_indices_ptrs = tl.make_block_ptr(
        base=perm_K_indices + pid_bz*stride_bz_perm_k_indices + pid_h*stride_h_perm_k_indices,
        shape=(tl.constexpr(1), kv_len),
        strides=(tl.constexpr(0), stride_seq_perm_k_indices),
        offsets=(0, 0),
        block_shape=(tl.constexpr(1), BLOCK_N),
        order=(1, 0)
    )
    perm_V_indices_ptrs = tl.make_block_ptr(
        base=perm_V_indices + pid_bz*stride_bz_perm_v_indices + pid_h*stride_h_perm_v_indices,
        shape=(kv_len, tl.constexpr(1)),
        strides=(stride_seq_perm_v_indices, tl.constexpr(0)),
        offsets=(0, 0),
        block_shape=(BLOCK_N, tl.constexpr(1)),
        order=(1, 0)
    )

    # Init accumulators
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    softmax_scale *= 1.44269504  # 1/log(2)
    # Load Q, iterate over K
    q = tl.load(Q_ptrs, boundary_check=(0,1), padding_option="zero")

    # Off-diagonal segments if causal
    acc, l_i, m_i = _permuted_block_sparse_attn_fwd_inner(
        acc, l_i, m_i,
        q,
        qo_len,
        kv_len,
        K_ptrs,
        V_ptrs, 
        K_block_indices_ptrs,
        perm_Q_indices_ptrs,
        perm_K_indices_ptrs,
        perm_V_indices_ptrs,
        stride_seq_k,
        stride_seq_v,
        stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
        stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
        stride_bz_perm_v_indices, stride_h_perm_v_indices, stride_seq_perm_v_indices,
        pid_seq,
        range_Q_seq,
        range_KV_seq,
        softmax_scale,
        dtype,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        SEGMENT_SIZE,
        LOGICAL_BLOCK_SIZE,
        4 - STAGE, 
    )

    if STAGE != 1: # causal==True
        # On-diagonal segments
        acc, l_i, m_i = _permuted_block_sparse_attn_fwd_inner(
            acc, l_i, m_i,
            q,
            qo_len,
            kv_len,
            K_ptrs,
            V_ptrs, 
            K_block_indices_ptrs,
            perm_Q_indices_ptrs,
            perm_K_indices_ptrs,
            perm_V_indices_ptrs,
            stride_seq_k,
            stride_seq_v,
            stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
            stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
            stride_bz_perm_v_indices, stride_h_perm_v_indices, stride_seq_perm_v_indices,
            pid_seq,
            range_Q_seq,
            range_KV_seq,
            softmax_scale,
            dtype,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            SEGMENT_SIZE,
            LOGICAL_BLOCK_SIZE,
            2, 
        )
    
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(dtype), boundary_check=(0,1))


def _permuted_block_sparse_attn_fwd_torch_naive(
        perm_Q, perm_K, perm_V, perm_O, # (batch_size, num_q_heads(num_kv_heads), q_len(kv_len), head_dim)
        perm_Q_indices, perm_K_indices, perm_V_indices, # (batch_size, num_q_heads, q_len(kv_len))
        block_mask,
        block_size=128,
        segment_size=1024,
        causal=True
):

    batch_size, num_q_heads, q_len, head_dim = perm_Q.shape
    batch_size, num_kv_heads, kv_len, head_dim = perm_K.shape
    assert num_q_heads == num_kv_heads
    assert causal
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_blocks = (kv_len + block_size - 1) // block_size
    # print(f"block_mask[0][0][0]: {block_mask[0][0][0]}")
    for b in range(batch_size):
        for h in range(num_q_heads):
            for q_block_idx in range(num_q_blocks):
                # Get query block
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, q_len)
                q_block = perm_Q[b, h, q_start:q_end, :]  # (block_size, head_dim)
                
                # Find selected key blocks
                selected_k_block_indices = torch.where(block_mask[b, h, q_block_idx, :] == True)[0]
                
                if len(selected_k_block_indices) == 0:
                    # No selected blocks, output zeros
                    perm_O[b, h, q_start:q_end, :] = 0
                    continue
                
                # Concatenate selected key tokens
                selected_k_blocks = []
                selected_v_blocks = []
                selected_k_indices_blocks = []
                for k_block_idx in selected_k_block_indices:
                    k_start = k_block_idx * block_size
                    k_end = min(k_start + block_size, kv_len)
                    k_block = perm_K[b, h, k_start:k_end, :]  # (block_size, head_dim)
                    v_block = perm_V[b, h, k_start:k_end, :]  # (block_size, head_dim)
                    k_indices_block = perm_K_indices[b, h, k_start:k_end]  # (block_size,)
                    selected_k_blocks.append(k_block)
                    selected_v_blocks.append(v_block)
                    selected_k_indices_blocks.append(k_indices_block)
                
                # Concatenate all selected blocks
                concat_k = torch.cat(selected_k_blocks, dim=0)  # (total_selected_tokens, head_dim)
                concat_v = torch.cat(selected_v_blocks, dim=0)  # (total_selected_tokens, head_dim)
                concat_k_indices = torch.cat(selected_k_indices_blocks, dim=0)  # (total_selected_tokens,)
                
                # Get query indices for causal masking
                q_indices = perm_Q_indices[b, h, q_start:q_end]  # (block_size,)
                
                # Compute attention scores
                scores = torch.matmul(q_block, concat_k.transpose(-2, -1))  # (block_size, total_selected_tokens)
                scores = scores / math.sqrt(head_dim)

                # Apply causal mask: query positions >= key positions
                causal_mask = q_indices.unsqueeze(-1) >= concat_k_indices.unsqueeze(0)  # (block_size, total_selected_tokens)
                scores = scores.masked_fill(~causal_mask, float('-inf'))
                
                # Apply softmax
                attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(perm_Q.dtype)  # (block_size, total_selected_tokens)
                
                # Compute output
                output_block = torch.matmul(attn_weights, concat_v)  # (block_size, head_dim)
                # Store output
                perm_O[b, h, q_start:q_end, :] = output_block

    return perm_O