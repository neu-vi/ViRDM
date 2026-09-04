from datetime import timedelta
from functools import partial
import os
from typing import Iterable
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def resolve_transformer_module(module_names):
    if module_names is None:
        return None

    from wan.modules.causal_model import CausalWanAttentionBlock

    name_to_module = {
        "causal_wan_block": CausalWanAttentionBlock,
    }
    if isinstance(module_names, str):
        module_names = [module_names]
    elif not isinstance(module_names, Iterable):
        raise TypeError(f"Unsupported transformer module spec: {type(module_names)!r}")

    resolved = set()
    for module_name in module_names:
        if isinstance(module_name, str):
            module_name = module_name.strip()
            if module_name not in name_to_module:
                raise ValueError(
                    f"Unknown transformer module '{module_name}'. "
                    f"Expected one of {sorted(name_to_module)}"
                )
            resolved.add(name_to_module[module_name])
        elif isinstance(module_name, type):
            resolved.add(module_name)
        else:
            raise TypeError(
                f"Unsupported transformer module entry: {type(module_name)!r}"
            )

    if not resolved:
        raise ValueError(
            "Transformer wrap strategy requires at least one transformer module"
        )
    return resolved


def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype or "float32").lower()
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }[dtype]


def get_fsdp_wrap_kwargs(
    config, prefix, default_transformer_modules=None, default_cpu_offload=False
):
    wrap_strategy = getattr(config, f"{prefix}_fsdp_wrap_strategy")
    kwargs = {
        "sharding_strategy": config.sharding_strategy,
        "mixed_precision": config.mixed_precision,
        "reduce_dtype": getattr(
            config,
            f"{prefix}_fsdp_reduce_dtype",
            getattr(config, "fsdp_reduce_dtype", "float32"),
        ),
        "buffer_dtype": getattr(
            config,
            f"{prefix}_fsdp_buffer_dtype",
            getattr(config, "fsdp_buffer_dtype", "float32"),
        ),
        "wrap_strategy": wrap_strategy,
        "min_num_params": int(
            getattr(config, f"{prefix}_fsdp_min_num_params", int(5e7))
        ),
        "cpu_offload": getattr(config, f"{prefix}_cpu_offload", default_cpu_offload),
    }
    if wrap_strategy == "transformer":
        transformer_modules = getattr(
            config,
            f"{prefix}_fsdp_transformer_modules",
            default_transformer_modules,
        )
        kwargs["transformer_module"] = resolve_transformer_module(transformer_modules)
    return kwargs


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    reduce_dtype="float32",
    buffer_dtype="float32",
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    cpu_offload=False,
    sync_module_states=False,
):
    if wrap_strategy == "none":
        return module

    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=_resolve_dtype(reduce_dtype),
            buffer_dtype=_resolve_dtype(buffer_dtype),
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=sync_module_states,
    )
    return module


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)


class _DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.local_rows = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        torch.cuda.current_stream().synchronize()
        dist.all_gather(gathered, tensor.contiguous())
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, gradient):
        start = ctx.rank * ctx.local_rows
        return gradient[start : start + ctx.local_rows].contiguous()


def differentiable_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return _DifferentiableAllGather.apply(tensor)
