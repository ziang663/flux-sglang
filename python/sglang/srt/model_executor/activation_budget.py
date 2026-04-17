from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


_BYTES_PER_GIB = 1 << 30


@dataclass(frozen=True)
class ActivationBudgetConfig:
    total_gpu_memory_gb: float
    mem_fraction_static: float
    hidden_size: int
    activation_dtype_size: int
    chunked_prefill_size: int
    max_prefill_tokens: int
    cuda_graph_max_bs: int
    disable_cuda_graph: bool
    piecewise_cuda_graph_max_tokens: int
    disable_piecewise_cuda_graph: bool
    tp_size: int
    ep_size: int
    moe_dp_size: int
    intermediate_size: Optional[int]
    moe_intermediate_size: Optional[int]
    num_experts_per_tok: Optional[int]
    avg_tokens_per_request: int
    attention_activation_factor: float
    graph_capture_token_factor: float
    safety_factor: float
    system_reserve_gb: float
    enabled: bool = True


@dataclass(frozen=True)
class ActivationBudgetResult:
    dynamic_headroom_bytes: int
    system_reserve_bytes: int
    graph_reserve_bytes: int
    activation_bytes_per_token: int
    max_running_tokens: int
    max_running_requests: int


def _ceil_div(numerator: int, denominator: int) -> int:
    return max((numerator + denominator - 1) // denominator, 1)


def _resolve_dense_intermediate_size(config: ActivationBudgetConfig) -> int:
    intermediate_size = config.intermediate_size
    if intermediate_size is None:
        intermediate_size = config.hidden_size * 4
    return _ceil_div(intermediate_size, max(config.tp_size, 1))


def _resolve_moe_intermediate_size(config: ActivationBudgetConfig) -> int:
    if config.moe_intermediate_size is None or config.num_experts_per_tok is None:
        return 0

    moe_parallel = max(config.ep_size * config.moe_dp_size, 1)
    moe_tp_size = max(config.tp_size // moe_parallel, 1)
    local_expert_width = _ceil_div(config.moe_intermediate_size, moe_tp_size)
    return local_expert_width * max(config.num_experts_per_tok, 1)


def estimate_activation_budget(
    config: ActivationBudgetConfig,
) -> Optional[ActivationBudgetResult]:
    if not config.enabled:
        return None

    dynamic_headroom_bytes = int(
        max(config.total_gpu_memory_gb * (1 - config.mem_fraction_static), 0)
        * _BYTES_PER_GIB
    )
    system_reserve_bytes = int(max(config.system_reserve_gb, 0) * _BYTES_PER_GIB)

    dense_activation_elems = config.hidden_size + _resolve_dense_intermediate_size(config)
    moe_activation_elems = config.hidden_size + _resolve_moe_intermediate_size(config)
    attn_activation_elems = int(
        math.ceil(config.hidden_size * max(config.attention_activation_factor, 0.0))
    )
    activation_bytes_per_token = int(
        math.ceil(
            max(dense_activation_elems, moe_activation_elems, attn_activation_elems)
            * config.activation_dtype_size
            * max(config.safety_factor, 1.0)
        )
    )

    graph_tokens = 0
    if not config.disable_cuda_graph:
        graph_tokens = max(graph_tokens, config.cuda_graph_max_bs)
    if not config.disable_piecewise_cuda_graph:
        graph_tokens = max(graph_tokens, config.piecewise_cuda_graph_max_tokens)

    graph_reserve_bytes = int(
        math.ceil(
            graph_tokens
            * activation_bytes_per_token
            * max(config.graph_capture_token_factor, 0.0)
        )
    )

    reserved_activation_bytes = system_reserve_bytes + graph_reserve_bytes
    if activation_bytes_per_token <= 0:
        max_running_tokens = 1
    else:
        max_running_tokens = max(
            (dynamic_headroom_bytes - reserved_activation_bytes)
            // activation_bytes_per_token,
            1,
        )

    avg_tokens_per_request = max(config.avg_tokens_per_request, 1)
    max_running_requests = max(max_running_tokens // avg_tokens_per_request, 1)

    return ActivationBudgetResult(
        dynamic_headroom_bytes=dynamic_headroom_bytes,
        system_reserve_bytes=system_reserve_bytes,
        graph_reserve_bytes=graph_reserve_bytes,
        activation_bytes_per_token=activation_bytes_per_token,
        max_running_tokens=max_running_tokens,
        max_running_requests=max_running_requests,
    )
