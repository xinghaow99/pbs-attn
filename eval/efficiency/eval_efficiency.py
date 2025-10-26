import os
import json
import time
import random
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import logging as hf_logging

from tqdm import tqdm

# Silence transformers warnings and reduce distributed verbosity
hf_logging.set_verbosity_error()
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

from pbs_attn.patch.huggingface import (
    apply_patch_with_prefill,
    get_flexprefill_prefill,
    get_minference_prefill,
    get_xattention_prefill,
    get_meanpooling_prefill,
    get_permuted_block_sparse_attn_fwd,
    get_flashattn_prefill,
)


@dataclass
class ScriptArguments:
    method: str = field(metadata={"help": "The method to benchmark."})
    len: int = field(default=16 * 1024, metadata={"help": "Length of the input sequence."})
    n_examples: int = field(default=10, metadata={"help": "Number of examples to average over."})
    num_warmup_iter: int = field(default=30, metadata={"help": "Number of warmup iterations."})
    no_save: bool = field(default=False, metadata={"help": "If enabled, do not save the result file."})
    output_dir: str = field(default="results", metadata={"help": "Directory to save results."})
    model_name: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "HF model id or local path to load (Llama, Qwen2, etc.)."},
    )

@dataclass
class FlexprefillArgs:
    gamma: float = field(default=0.95, metadata={"help": "Gamma for Flexprefill."})
    tau: float = field(default=0.1, metadata={"help": "Tau for Flexprefill."})


@dataclass
class XattnArgs:
    stride: int = field(default=8, metadata={"help": "Stride for XAttention."})
    threshold: float = field(default=0.9, metadata={"help": "Threshold for XAttention."})
    block_size: int = field(default=128, metadata={"help": "Block size for XAttention."})
    keep_sink: bool = field(default=True, metadata={"help": "Keep sink tokens in XAttention."})
    keep_recent: bool = field(default=True, metadata={"help": "Keep recent tokens in XAttention."})


@dataclass
class MeanPoolingArgs:
    block_size: int = field(default=128, metadata={"help": "Block size for MeanPooling."})
    segment_size: int = field(default=1024, metadata={"help": "Segment size for MeanPooling."})
    threshold: float = field(default=0.9, metadata={"help": "Threshold for MeanPooling."})
    force_select_first_block: bool = field(default=True, metadata={"help": "Force select first block for MeanPooling."})
    force_select_current_block: bool = field(default=True, metadata={"help": "Force select current block for MeanPooling."})

@dataclass
class PBSArgs:
    block_size: int = field(default=128, metadata={"help": "Block size for PBS-Attn."})
    segment_size: int = field(default=256, metadata={"help": "Segment size for PBS-Attn."})
    threshold: float = field(default=0.9, metadata={"help": "Threshold for PBS-Attn."})
    force_select_first_block: bool = field(default=True, metadata={"help": "Force select first block for PBS-Attn."})

def build_prefill_fn(method: str, method_args):
    if method == "flexprefill":
        args = method_args or FlexprefillArgs()
        return get_flexprefill_prefill(gamma=args.gamma, tau=args.tau)
    if method == "minference":
        return get_minference_prefill()
    if method == "flashattn":
        return get_flashattn_prefill()
    if method == "xattention":
        args = method_args or XattnArgs()
        return get_xattention_prefill(
            stride=args.stride,
            threshold=args.threshold,
            block_size=args.block_size,
            keep_sink=args.keep_sink,
            keep_recent=args.keep_recent,
        )
    if method == "meanpooling":
        args = method_args or MeanPoolingArgs()
        return get_meanpooling_prefill(
            block_size=args.block_size,
            segment_size=args.segment_size,
            threshold=args.threshold,
            force_select_first_block=args.force_select_first_block,
            force_select_current_block=args.force_select_current_block
        )
    if method == "pbs":
        args = method_args or PBSArgs()
        return get_permuted_block_sparse_attn_fwd(
            block_size=args.block_size,
            segment_size=args.segment_size,
            threshold=args.threshold,
            force_select_first_block=args.force_select_first_block,
            use_triton=True,
        )
    raise NotImplementedError(f"Unknown method: {method}")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def init_distributed_if_needed():
    if dist.is_available() and not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group(backend="nccl")


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def warmup_generate(model, tokenizer, length: int, num_iter: int = 10):
    
    vocab_size = model.get_input_embeddings().weight.size(0)
    input_ids = torch.randint(0, vocab_size, (1, length), device=model.device, dtype=torch.long)

    with torch.no_grad():
        for _ in tqdm(range(num_iter), desc="Warmup", disable=get_rank() != 0):
            # Warm up using generate to exercise the patched prefill path
            _ = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)


def measure_prefill_latency(model, input_ids: torch.Tensor) -> float:
    # Measure E2E prefill using a single-token generate call
    e2e_start = torch.cuda.Event(enable_timing=True)
    e2e_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e2e_start.record()
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)
    e2e_end.record()
    torch.cuda.synchronize()
    return e2e_start.elapsed_time(e2e_end) / 1000.0


def main(script_args, method_args):
    # Seed
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Distributed setup
    init_distributed_if_needed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Load dataset
    if get_rank() == 0:
        print("Loading LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    long_examples = [d for d in dataset if d['length'] == 'long']

    if len(long_examples) < script_args.n_examples:
        if get_rank() == 0:
            print(f"Warning: Requested {script_args.n_examples} examples, but only {len(long_examples)} long examples found. Using all available.")
        n_examples = len(long_examples)
        sampled_examples = long_examples
    else:
        n_examples = script_args.n_examples
        sampled_examples = random.sample(long_examples, n_examples)

    # Load model/tokenizer
    model_name = script_args.model_name
    if get_rank() == 0:
        print(f"Loading model and tokenizer for {model_name}...")

    # Default tokenizer derived from model name
    tokenizer_id = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    from_pretrained_kwargs = {"torch_dtype": torch.bfloat16}
    # Enable TP plan if running under torchrun; rely on env-specific support
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from_pretrained_kwargs["tp_plan"] = "auto"
    else:
        from_pretrained_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs)

    prefill_fn = build_prefill_fn(script_args.method, method_args)
    model = apply_patch_with_prefill(model, prefill_fn)

    # Warmup
    if get_rank() == 0:
        print("Warming up...")
    if is_distributed():
        dist.barrier()
    warmup_generate(model, tokenizer, length=script_args.len, num_iter=script_args.num_warmup_iter)

    # Benchmark
    e2e_latencies = []
    # attn_latencies = []  # Not directly measurable here; keep NaN for compatibility
    num_tokens = 0

    if get_rank() == 0:
        print(f"Starting benchmark over {n_examples} examples with method: {script_args.method}...")
    examples_iterable = long_examples if n_examples == len(long_examples) else sampled_examples
    for example in tqdm(examples_iterable, total=n_examples, desc="Benchmark", disable=get_rank() != 0):
        context = example['context']
        inputs = tokenizer(context, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        if script_args.len < input_ids.shape[1]:
            input_ids = input_ids[:, :script_args.len]
        else:
            input_ids = input_ids.repeat(1, script_args.len // input_ids.shape[1] + 1)[:, :script_args.len]

        num_tokens = input_ids.shape[1]

        # Measure prefill latency
        if is_distributed():
            dist.barrier()
        e2e_latency = measure_prefill_latency(model, input_ids)
        if is_distributed():
            tensor_latency = torch.tensor([e2e_latency], device=model.device, dtype=torch.float32)
            dist.all_reduce(tensor_latency, op=dist.ReduceOp.MAX)
            e2e_latency = float(tensor_latency.item())
        e2e_latencies.append(e2e_latency)
        with torch.no_grad():
            _ = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)

    # Aggregate results
    avg_e2e_latency = sum(e2e_latencies) / len(e2e_latencies)
    if get_rank() == 0:
        print(f"\n--- Averaged Results over {n_examples} examples ---")
        print(f"Number of tokens: {num_tokens}")
        print(f"Average E2E Prefill latency: {avg_e2e_latency:.6f} seconds")

    if len(e2e_latencies) > 1:
        e2e_std = torch.tensor(e2e_latencies).std().item()
        if get_rank() == 0:
            print(f"E2E Prefill latency std dev: {e2e_std:.6f} seconds")
    else:
        e2e_std = None

    if get_rank() == 0 and not script_args.no_save:
        if os.path.isabs(script_args.output_dir):
            results_dir = script_args.output_dir
        else:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_args.output_dir)
        os.makedirs(results_dir, exist_ok=True)

        results = {
            "script_args": script_args.__dict__,
            "method_args": method_args.__dict__ if method_args else {},
            "model_name": model_name,
            "tokenizer_name": tokenizer_id,
            "num_tokens": num_tokens,
            "avg_e2e_latency": avg_e2e_latency,
        }
        if e2e_std is not None:
            results["e2e_std"] = e2e_std

        method_args_str = ""
        if method_args:
            items = []
            for key, value in method_args.__dict__.items():
                fname_key = key
                items.append(f"{fname_key}={value}")
            method_args_str = "_" + "_".join(items)

        file_path = os.path.join(results_dir, f"{script_args.method}_{script_args.len}.json")
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {file_path}")


    # Clean up distributed state
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    method_class_map = {
        "flexprefill": FlexprefillArgs,
        "xattention": XattnArgs,
        "pbs": PBSArgs,
        "meanpooling": MeanPoolingArgs,
    }

    method_args = None
    if script_args.method in method_class_map:
        method_class = method_class_map[script_args.method]
        sub_parser = HfArgumentParser(method_class)
        method_args = sub_parser.parse_args_into_dataclasses(args=remaining_args)[0]

    main(script_args, method_args)


