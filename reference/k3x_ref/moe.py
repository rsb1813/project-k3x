# K3 router와 native MXFP4 Stable LatentMoE 경로를 구현합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from k3x_ref.config import SyntheticK3Config
from k3x_ref.mxfp4 import mxfp4_matmul
from k3x_ref.ops import rms_norm, situ_glu


@dataclass(frozen=True)
class RouterOutput:
    expert_ids: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class PackedMatrix:
    packed: bytes
    scales: bytes
    rows: int
    cols: int

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        return mxfp4_matmul(x, self.packed, self.scales, self.rows, self.cols)


@dataclass(frozen=True)
class ExpertWeights:
    gate: PackedMatrix
    up: PackedMatrix
    down: PackedMatrix


@dataclass(frozen=True)
class LatentMoEWeights:
    router_weight: torch.Tensor
    correction_bias: torch.Tensor
    routed_down_proj: torch.Tensor
    routed_up_proj: torch.Tensor
    routed_norm: torch.Tensor
    experts: tuple[ExpertWeights, ...]
    shared_gate: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor


def route(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    routed_scale: float,
) -> RouterOutput:
    scores = torch.sigmoid(
        functional.linear(hidden.to(torch.float32), weight.to(torch.float32))
    )
    adjusted = scores + correction_bias.to(device=hidden.device, dtype=torch.float32)
    expert_ids = torch.argsort(adjusted, dim=-1, descending=True, stable=True)[..., :top_k]
    selected = torch.gather(scores, -1, expert_ids)
    normalized = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    return RouterOutput(expert_ids, normalized * routed_scale, scores)


def _expert_forward(
    hidden: torch.Tensor,
    weights: ExpertWeights,
    cfg: SyntheticK3Config,
) -> torch.Tensor:
    gate = weights.gate.matmul(hidden)
    up = weights.up.matmul(hidden)
    activated = situ_glu(
        gate,
        up,
        cfg.activation_situ_beta,
        cfg.activation_situ_linear_beta,
    )
    return weights.down.matmul(activated)


def stable_latent_moe(
    hidden: torch.Tensor,
    weights: LatentMoEWeights,
    cfg: SyntheticK3Config,
) -> torch.Tensor:
    if len(weights.experts) != cfg.num_experts:
        raise ValueError("experts must match num_experts")
    leading_shape = hidden.shape[:-1]
    flat_hidden = hidden.reshape(-1, cfg.hidden_size)
    routed = route(
        flat_hidden,
        weights.router_weight,
        weights.correction_bias,
        cfg.top_k,
        cfg.routed_scaling_factor,
    )
    latent = functional.linear(flat_hidden, weights.routed_down_proj)
    mixed = torch.zeros_like(latent, dtype=torch.float32)
    for token_index in range(flat_hidden.shape[0]):
        for slot in range(cfg.top_k):
            expert_id = int(routed.expert_ids[token_index, slot])
            expert_output = _expert_forward(
                latent[token_index : token_index + 1],
                weights.experts[expert_id],
                cfg,
            )
            mixed[token_index] += routed.weights[token_index, slot] * expert_output[0]
    routed_hidden = functional.linear(
        rms_norm(mixed, weights.routed_norm, cfg.rms_norm_eps),
        weights.routed_up_proj,
    )
    shared_gate = functional.linear(flat_hidden, weights.shared_gate)
    shared_up = functional.linear(flat_hidden, weights.shared_up)
    shared = functional.linear(
        situ_glu(
            shared_gate,
            shared_up,
            cfg.activation_situ_beta,
            cfg.activation_situ_linear_beta,
        ),
        weights.shared_down,
    )
    return (routed_hidden + shared).reshape(*leading_shape, cfg.hidden_size)

