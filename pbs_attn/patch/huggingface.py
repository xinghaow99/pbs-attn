      
# coding=utf-8
import torch
import torch.nn.functional as F
import types
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.cache_utils import Cache
from typing import Optional, Tuple, Callable


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# =================================================================================
# >> PREFILL KERNEL CONFIGURATIONS
# =================================================================================

def get_meanpooling_prefill(block_size=128, segment_size=1024, threshold=0.9, 
                           force_select_first_block=True, force_select_current_block=True,):
    """Configure MeanPooling prefill kernel with specific parameters."""
    try:
        from pbs_attn.baselines.MeanPooling import MeanPooling_prefill as _MeanPooling_prefill
    except Exception as e:
        raise RuntimeError("MeanPooling unavailable.") from e
    return partial(
        _MeanPooling_prefill,
        block_size=block_size,
        segment_size=segment_size,
        threshold=threshold,
        causal=True,
        force_select_first_block=force_select_first_block,
        force_select_current_block=force_select_current_block,
    )

def get_minference_prefill(vertical_size=1000, slash_size=6096, adaptive_budget=None):
    """Configure Minference prefill kernel with specific parameters."""
    try:
        from pbs_attn.baselines.Minference import Minference_prefill as _Minference_prefill
    except NameError as e:
        if "name 'Tuple' is not defined" in str(e):
            import importlib.util
            import sys
            import typing
            module_name = "minference.ops.pit_sparse_flash_attention_v2"
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                raise
            minference_module = importlib.util.module_from_spec(spec)
            minference_module.Tuple = typing.Tuple
            sys.modules[module_name] = minference_module
            spec.loader.exec_module(minference_module)
            from pbs_attn.baselines.Minference import Minference_prefill as _Minference_prefill
        else:
            raise
    except Exception as e:
        raise RuntimeError("Minference unavailable.") from e
    return partial(
        _Minference_prefill,
        vertical_size=vertical_size,
        slash_size=slash_size,
        adaptive_budget=adaptive_budget
    )


def get_xattention_prefill(stride, norm=1, threshold=0.8, block_size=128, use_triton=True,
                          kdb=1, chunk_size=None, keep_sink=False, keep_recent=False):
    """Configure XAttention prefill kernel with specific parameters."""
    try:
        from pbs_attn.baselines.XAttention import Xattention_prefill as _Xattention_prefill
    except Exception as e:
        raise RuntimeError("XAttention unavailable.") from e
    return partial(
        _Xattention_prefill,
        stride=stride,
        norm=norm,
        threshold=threshold,
        block_size=block_size,
        use_triton=use_triton,
        causal=True,
        kdb=kdb,
        chunk_size=chunk_size,
        keep_sink=keep_sink,
        keep_recent=keep_recent
    )

def get_flexprefill_prefill(gamma=0.9, tau=0, min_budget=None, max_budget=None, 
                           gqa_interleave=False, softmax_scale=None, block_size=128):
    """Configure FlexPrefill kernel with specific parameters."""
    try:
        from pbs_attn.baselines.Flexprefill import Flexprefill_prefill as _Flexprefill_prefill
    except Exception as e:
        raise RuntimeError("FlexPrefill unavailable.") from e
    return partial(
        _Flexprefill_prefill,
        gamma=gamma,
        tau=tau,
        min_budget=min_budget,
        max_budget=max_budget,
        gqa_interleave=gqa_interleave,
        softmax_scale=softmax_scale,
        block_size=block_size
    )

def get_flashattn_prefill(causal=True):
    """Configure FlashAttn prefill kernel."""
    try:
        from pbs_attn.baselines.FlashAttn import FlashAttn_prefill as _FlashAttn_prefill
    except Exception as e:
        raise RuntimeError("FlashAttn unavailable.") from e
    return partial(
        _FlashAttn_prefill,
        causal=causal
    )


def get_permuted_block_sparse_attn_fwd(
    block_size=128,
    segment_size=256,
    threshold=0.9,
    causal=True,
    force_select_first_block=True,
    use_triton=True,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
):
    """Configure Permuted Block Sparse Attention forward kernel."""
    try:
        from pbs_attn.src.pbs import permuted_block_sparse_attn_fwd as _pbs_impl
    except Exception as e:
        raise RuntimeError("PBS unavailable.") from e

    return partial(
        _pbs_impl,
        block_size=block_size,
        segment_size=segment_size,
        threshold=threshold,
        causal=causal,
        force_select_first_block=force_select_first_block,
        use_triton=use_triton,
        query_pool_mode=query_pool_mode,
        key_pool_mode=key_pool_mode,
    )

def patched_attention_forward(
    self,  # Union[LlamaAttention, Qwen2Attention]
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    prefill_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    This function replaces `LlamaAttention.forward` and `Qwen2Attention.forward`.
    It routes to the specified prefill function for prompts and uses the original method for decoding.
    
    Args:
        prefill_fn: A callable that takes (query_states, key_states, value_states) and returns attention output.
                   Can be configured using partial functions for different kernels.
    """
    bsz, q_len, hidden_dim = hidden_states.size()

    if q_len > 1:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape for multi-head attention (supporting tensor parallel local shards)
        local_num_q_heads = query_states.shape[-1] // self.head_dim
        local_num_kv_heads = key_states.shape[-1] // self.head_dim
        query_states = query_states.view(bsz, q_len, local_num_q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, local_num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, local_num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Position Embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat kv to match q heads locally (GQA)
        kv_repeat = max(1, local_num_q_heads // max(1, local_num_kv_heads))
        key_states = repeat_kv(key_states, kv_repeat)
        value_states = repeat_kv(value_states, kv_repeat)

        # 4. Call the configured prefill kernel
        if prefill_fn is None:
            raise RuntimeError("prefill_fn required.")
        else:
            # For kernels that require layer_idx, add it to the call
            kernel_fn = prefill_fn.func if isinstance(prefill_fn, partial) else prefill_fn
            if 'layer_idx' in kernel_fn.__code__.co_varnames:
                attn_output = prefill_fn(query_states, key_states, value_states, layer_idx=self.layer_idx)
            else:
                attn_output = prefill_fn(query_states, key_states, value_states)

        # reshape to local tensor-parallel hidden dim (num_local_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, local_num_q_heads * self.head_dim)

        # 5. Apply the final output projection (sharded RowParallelLinear under TP)
        attn_output = self.o_proj(attn_output)

        return attn_output, None
    else:
        # --- DECODE PATH ---
        # Use the original, highly-optimized forward method for single-token generation.
        return self.original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs
        )

def apply_patch_with_prefill(model, prefill_fn: Optional[Callable] = None, skip_layers: Optional[list] = None):
    """
    Apply the monkey patch to all self-attention layers with a specific prefill function.
    Supports both Llama and Qwen models.
    
    Args:
        model: The model to patch (LlamaForCausalLM or Qwen2ForCausalLM)
        prefill_fn: A callable configured with partial() for the specific kernel and parameters
        skip_layers: A list of layer indices to skip when applying the patch (0-indexed)
    """
    if skip_layers is None:
        skip_layers = []
    
    # Validate model type
    if not is_model_supported(model):
        supported_types = get_supported_model_types()
        raise ValueError(f"Model type {type(model).__name__} is not supported. Supported types: {supported_types}")
    
    # Detect model type
    model_type = type(model).__name__
    attention_type = type(model.model.layers[0].self_attn).__name__ if len(model.model.layers) > 0 else "Unknown"
    
    print(f"\nApplying monkey patch to {model_type} with {attention_type}")
    print(f"Prefill function: {getattr(prefill_fn, 'func', prefill_fn).__name__ if prefill_fn else 'MeanPooling_prefill'}")
    if skip_layers:
        print(f"Skipping layers: {skip_layers}")
    
    patched_count = 0
    total_layers = len(model.model.layers)
    
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in skip_layers:
            print(f"  Skipping layer {layer_idx}")
            continue
            
        # Store the original forward method
        layer.self_attn.original_forward = layer.self_attn.forward
        
        # This wrapper will become the new forward method.
        def new_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            return patched_attention_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                prefill_fn=prefill_fn,
                **kwargs
            )
        
        # Apply the patch
        layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)
        patched_count += 1
    
    print(f"✅ Monkey patch applied successfully. Patched {patched_count}/{total_layers} layers.")
    return model

def get_supported_model_types():
    """Return list of supported model types for patching."""
    return ["LlamaForCausalLM", "Qwen2ForCausalLM"]

def is_model_supported(model):
    """Check if a model is supported for patching."""
    model_type = type(model).__name__
    return model_type in get_supported_model_types()

# =================================================================================
# >> MAIN TESTING SCRIPT
# =================================================================================

def main():
    """Main function to load a Llama-style or Qwen model, apply the patch, and run a test."""
    # Choose model: Llama or Qwen
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading model and tokenizer for '{model_name}'...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
        # tp_plan='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


    # Running without patch
    print("\nRunning `model.generate()` without the patched attention method...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    print("\n--- Final Output ---")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("--------------------")

    # Example 1: Using MeanPooling with custom parameters
    # print("\n=== Example 1: MeanPooling Prefill ===")
    # meanpooling_fn = get_meanpooling_prefill(
    #     block_size=128,
    #     segment_size=1024,
    #     threshold=0.9,
    # )
    # model = apply_patch_with_prefill(model, meanpooling_fn)

    # Example 2: Using Minference
    # print("\n=== Example 2: Minference Prefill ===")
    # minference_fn = get_minference_prefill(
    #     vertical_size=500,
    #     slash_size=3000,
    #     adaptive_budget=0.1  # 10% of sequence length
    # )
    # model = apply_patch_with_prefill(model, minference_fn)

    # # Example 3: Using XAttention
    # print("\n=== Example 4: XAttention Prefill ===")
    # xattention_fn = get_xattention_prefill(
    #     stride=8,
    #     threshold=0.9,
    #     block_size=128
    # )
    # model = apply_patch_with_prefill(model, xattention_fn)

    # # Example 4: Using FlexPrefill (works with both Llama and Qwen models)
    # print("\n=== Example 5: FlexPrefill ===")
    # flexprefill_fn = get_flexprefill_prefill(
    #     gamma=0.95,
    #     tau=0.1,
    # )
    # model = apply_patch_with_prefill(model, flexprefill_fn)

    # Example 5: Using FlashAttn
    # print("\n=== Example 6: FlashAttn ===")
    # flashattn_fn = get_flashattn_prefill()
    # model = apply_patch_with_prefill(model, flashattn_fn)

    # Example 6: Using Permuted Block Sparse Attention
    print("\n=== Example 7: Permuted Block Sparse Attention ===")
    pbs_fn = get_permuted_block_sparse_attn_fwd(
        block_size=128,
        segment_size=256,
        threshold=0.9,
        force_select_first_block=True,
        use_triton=True
    )
    model = apply_patch_with_prefill(model, pbs_fn)

    print("\nRunning `model.generate()` with the patched attention method...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    print("\n--- Final Output ---")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("--------------------")

if __name__ == "__main__":
    main()

    