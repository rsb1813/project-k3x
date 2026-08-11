# 공식 KDA의 BF16 경계와 V-first recurrent state를 검증하는 독립 scalar oracle입니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class OfficialKdaConfig:
    hidden_size: int
    heads: int
    head_dim: int
    conv_width: int
    rms_norm_epsilon: float
    gate_lower_bound: float

    @property
    def projection_size(self) -> int:
        return self.heads * self.head_dim


@dataclass(frozen=True)
class OfficialKdaWeights:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    f_a_proj: torch.Tensor
    f_b_proj: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    b_proj: torch.Tensor
    g_proj: torch.Tensor
    o_norm: torch.Tensor
    o_proj: torch.Tensor


@dataclass(frozen=True)
class OfficialKdaState:
    conv_q: torch.Tensor
    conv_k: torch.Tensor
    conv_v: torch.Tensor
    recurrent_v_first: torch.Tensor


@dataclass(frozen=True)
class OfficialKdaBoundaries:
    projected_q: torch.Tensor
    projected_k: torch.Tensor
    projected_v: torch.Tensor
    convolved_q: torch.Tensor
    convolved_k: torch.Tensor
    convolved_v: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    recurrent_output: torch.Tensor
    gated: torch.Tensor


@dataclass(frozen=True)
class OfficialKdaResult:
    output: torch.Tensor
    state: OfficialKdaState
    boundaries: OfficialKdaBoundaries


def _invalid(message: str) -> ValueError:
    return ValueError(f"invalid official KDA: {message}")


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise _invalid(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise _invalid(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise _invalid(f"{name} device must be {device}, got {tensor.device}")
    if not torch.isfinite(tensor).all():
        raise _invalid(f"{name} contains a non-finite value")


def _validate_config(config: OfficialKdaConfig) -> None:
    if min(config.hidden_size, config.heads, config.head_dim) <= 0:
        raise _invalid("hidden_size, heads, and head_dim must be positive")
    if config.conv_width < 2:
        raise _invalid("conv_width must be at least two")
    if config.rms_norm_epsilon <= 0:
        raise _invalid("rms_norm_epsilon must be positive")
    if config.gate_lower_bound >= 0:
        raise _invalid("gate_lower_bound must be negative")


def _validate_inputs(
    hidden: torch.Tensor,
    weights: OfficialKdaWeights,
    state: OfficialKdaState,
    config: OfficialKdaConfig,
) -> None:
    _validate_config(config)
    if hidden.ndim != 3:
        raise _invalid("hidden must have shape [batch, sequence, hidden_size]")
    batch, sequence, hidden_size = hidden.shape
    if batch <= 0 or sequence <= 0:
        raise _invalid("batch and sequence dimensions must be positive")
    if hidden_size != config.hidden_size:
        raise _invalid("hidden width does not match hidden_size")
    if hidden.dtype != torch.bfloat16:
        raise _invalid("hidden dtype must be torch.bfloat16")
    if not torch.isfinite(hidden).all():
        raise _invalid("hidden contains a non-finite value")

    device = hidden.device
    projection = config.projection_size
    bf16_shapes = {
        "q_proj": (projection, config.hidden_size),
        "k_proj": (projection, config.hidden_size),
        "v_proj": (projection, config.hidden_size),
        "f_a_proj": (config.head_dim, config.hidden_size),
        "f_b_proj": (projection, config.head_dim),
        "b_proj": (config.heads, config.hidden_size),
        "g_proj": (projection, config.hidden_size),
        "o_proj": (config.hidden_size, projection),
    }
    for name, shape in bf16_shapes.items():
        _require_tensor(name, getattr(weights, name), shape, torch.bfloat16, device)

    fp32_shapes = {
        "q_conv": (projection, config.conv_width),
        "k_conv": (projection, config.conv_width),
        "v_conv": (projection, config.conv_width),
        "a_log": (config.head_dim,),
        "dt_bias": (projection,),
        "o_norm": (config.head_dim,),
    }
    for name, shape in fp32_shapes.items():
        _require_tensor(name, getattr(weights, name), shape, torch.float32, device)

    history_shape = (batch, config.conv_width - 1, projection)
    _require_tensor("conv_q", state.conv_q, history_shape, torch.bfloat16, device)
    _require_tensor("conv_k", state.conv_k, history_shape, torch.bfloat16, device)
    _require_tensor("conv_v", state.conv_v, history_shape, torch.bfloat16, device)
    _require_tensor(
        "recurrent_v_first",
        state.recurrent_v_first,
        (batch, config.heads, config.head_dim, config.head_dim),
        torch.float32,
        device,
    )


def zero_official_kda_state(
    config: OfficialKdaConfig,
    batch_size: int,
    device: torch.device,
) -> OfficialKdaState:
    _validate_config(config)
    if batch_size <= 0:
        raise _invalid("batch_size must be positive")
    history_shape = (
        batch_size,
        config.conv_width - 1,
        config.projection_size,
    )
    recurrent_shape = (
        batch_size,
        config.heads,
        config.head_dim,
        config.head_dim,
    )
    return OfficialKdaState(
        conv_q=torch.zeros(history_shape, dtype=torch.bfloat16, device=device),
        conv_k=torch.zeros(history_shape, dtype=torch.bfloat16, device=device),
        conv_v=torch.zeros(history_shape, dtype=torch.bfloat16, device=device),
        recurrent_v_first=torch.zeros(
            recurrent_shape, dtype=torch.float32, device=device
        ),
    )


def _project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return functional.linear(hidden, weight).to(torch.bfloat16)


def _short_conv(
    projected: torch.Tensor,
    history: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    current = history
    transposed_weight = weight.transpose(0, 1)
    for index in range(projected.shape[1]):
        window = torch.cat((current, projected[:, index : index + 1]), dim=1)
        convolved = (window.float() * transposed_weight.unsqueeze(0)).sum(dim=1)
        outputs.append(functional.silu(convolved).to(torch.bfloat16))
        current = window[:, 1:].to(torch.bfloat16)
    return torch.stack(outputs, dim=1), current


def _normalize_heads(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    normalized = functional.normalize(value.float(), p=2.0, dim=-1) * scale
    return normalized.to(torch.bfloat16)


def official_kda(
    hidden: torch.Tensor,
    weights: OfficialKdaWeights,
    state: OfficialKdaState,
    config: OfficialKdaConfig,
) -> OfficialKdaResult:
    _validate_inputs(hidden, weights, state, config)
    batch, sequence, _ = hidden.shape
    projection = config.projection_size
    head_shape = (batch, sequence, config.heads, config.head_dim)

    projected_q = _project(hidden, weights.q_proj)
    projected_k = _project(hidden, weights.k_proj)
    projected_v = _project(hidden, weights.v_proj)
    convolved_q, conv_q = _short_conv(projected_q, state.conv_q, weights.q_conv)
    convolved_k, conv_k = _short_conv(projected_k, state.conv_k, weights.k_conv)
    convolved_v, conv_v = _short_conv(projected_v, state.conv_v, weights.v_conv)

    q = _normalize_heads(
        convolved_q.reshape(head_shape), config.head_dim**-0.5
    )
    k = _normalize_heads(convolved_k.reshape(head_shape))
    v = convolved_v.reshape(head_shape)

    forget_low_rank = _project(hidden, weights.f_a_proj)
    forget = _project(forget_low_rank, weights.f_b_proj).reshape(head_shape)
    decay_argument = forget.float() + weights.dt_bias.reshape(
        1, 1, config.heads, config.head_dim
    )
    log_decay = config.gate_lower_bound * torch.sigmoid(
        weights.a_log.exp().reshape(1, 1, 1, config.head_dim) * decay_argument
    )
    beta = torch.sigmoid(_project(hidden, weights.b_proj).float())

    current_kv = state.recurrent_v_first.transpose(-1, -2).clone()
    recurrent_outputs: list[torch.Tensor] = []
    for index in range(sequence):
        alpha = log_decay[:, index].exp()
        decayed = alpha.unsqueeze(-1) * current_kv
        key = k[:, index].float()
        value = v[:, index].float()
        prediction = torch.einsum("bhk,bhkv->bhv", key, decayed)
        delta = (value - prediction) * beta[:, index].unsqueeze(-1)
        current_kv = decayed + key.unsqueeze(-1) * delta.unsqueeze(-2)
        recurrent_outputs.append(
            torch.einsum("bhk,bhkv->bhv", q[:, index].float(), current_kv)
        )
    recurrent_output = torch.stack(recurrent_outputs, dim=1)

    mean_square = recurrent_output.square().mean(dim=-1, keepdim=True)
    normalized = recurrent_output * torch.rsqrt(
        mean_square + config.rms_norm_epsilon
    )
    normalized = normalized * weights.o_norm.reshape(1, 1, 1, config.head_dim)
    output_gate = torch.sigmoid(
        _project(hidden, weights.g_proj).reshape(head_shape).float()
    )
    gated = (normalized * output_gate).to(torch.bfloat16)
    output = _project(gated.reshape(batch, sequence, projection), weights.o_proj)

    final_state = OfficialKdaState(
        conv_q=conv_q,
        conv_k=conv_k,
        conv_v=conv_v,
        recurrent_v_first=current_kv.transpose(-1, -2),
    )
    boundaries = OfficialKdaBoundaries(
        projected_q=projected_q,
        projected_k=projected_k,
        projected_v=projected_v,
        convolved_q=convolved_q,
        convolved_k=convolved_k,
        convolved_v=convolved_v,
        q=q,
        k=k,
        v=v,
        log_decay=log_decay,
        beta=beta,
        recurrent_output=recurrent_output,
        gated=gated,
    )
    return OfficialKdaResult(output=output, state=final_state, boundaries=boundaries)
