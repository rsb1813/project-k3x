# 공식 KDA scalar oracle의 full/incremental 상태와 channel-wise recurrence를 검증합니다.
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from k3x_ref.official_kda import (
    OfficialKdaConfig,
    OfficialKdaState,
    OfficialKdaWeights,
    official_kda,
    zero_official_kda_state,
)
from k3x_ref.official_kda import _project


def _config() -> OfficialKdaConfig:
    return OfficialKdaConfig(
        hidden_size=4,
        heads=2,
        head_dim=2,
        conv_width=3,
        rms_norm_epsilon=1.0e-5,
        gate_lower_bound=-5.0,
    )


def _weights() -> OfficialKdaWeights:
    identity = torch.eye(4, dtype=torch.bfloat16)
    conv = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [-0.25, 0.5, 0.75],
            [0.5, 0.25, -0.5],
            [-0.5, 0.25, 1.25],
        ],
        dtype=torch.float32,
    )
    return OfficialKdaWeights(
        q_proj=identity,
        k_proj=torch.tensor(
            [
                [0.5, 0.0, 0.25, 0.0],
                [0.0, 0.75, 0.0, -0.25],
                [0.25, 0.0, 1.0, 0.0],
                [0.0, -0.5, 0.0, 0.5],
            ],
            dtype=torch.bfloat16,
        ),
        v_proj=identity,
        q_conv=conv,
        k_conv=conv * 0.75,
        v_conv=conv * -0.5,
        f_a_proj=torch.tensor(
            [[0.5, -0.25, 0.0, 0.25], [0.0, 0.5, -0.5, 0.25]],
            dtype=torch.bfloat16,
        ),
        f_b_proj=torch.tensor(
            [[1.0, 0.0], [0.5, -0.5], [0.0, 1.0], [-0.25, 0.75]],
            dtype=torch.bfloat16,
        ),
        a_log=torch.tensor([0.0, 0.5], dtype=torch.float32),
        dt_bias=torch.tensor([0.25, -0.5, 0.75, -0.25], dtype=torch.float32),
        b_proj=torch.tensor(
            [[0.5, -0.25, 0.25, 0.0], [0.0, 0.5, -0.5, 0.25]],
            dtype=torch.bfloat16,
        ),
        g_proj=torch.tensor(
            [
                [0.5, 0.0, 0.0, -0.25],
                [0.0, 0.5, 0.25, 0.0],
                [-0.25, 0.0, 0.5, 0.0],
                [0.0, 0.25, 0.0, 0.75],
            ],
            dtype=torch.bfloat16,
        ),
        o_norm=torch.tensor([1.0, 1.5], dtype=torch.float32),
        o_proj=identity,
    )


def _tokens() -> torch.Tensor:
    return torch.tensor(
        [[[0.5, -1.0, 0.25, 0.75], [-0.25, 0.5, 1.0, -0.5]]],
        dtype=torch.bfloat16,
    )


def test_official_kda_full_and_incremental_paths_match_all_final_state() -> None:
    cfg = _config()
    weights = _weights()
    tokens = _tokens()
    zero = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)

    full = official_kda(tokens, weights, zero, cfg)
    first = official_kda(tokens[:, :1], weights, zero, cfg)
    second = official_kda(tokens[:, 1:], weights, first.state, cfg)

    torch.testing.assert_close(
        full.output,
        torch.cat((first.output, second.output), dim=1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(full.state.conv_q, second.state.conv_q, atol=0, rtol=0)
    torch.testing.assert_close(full.state.conv_k, second.state.conv_k, atol=0, rtol=0)
    torch.testing.assert_close(full.state.conv_v, second.state.conv_v, atol=0, rtol=0)
    torch.testing.assert_close(
        full.state.recurrent_v_first,
        second.state.recurrent_v_first,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert full.output.dtype == torch.bfloat16
    assert full.state.conv_q.dtype == torch.bfloat16
    assert full.state.recurrent_v_first.dtype == torch.float32
    assert full.boundaries.q.shape == (1, 2, 2, 2)
    assert full.boundaries.log_decay.shape == (1, 2, 2, 2)
    assert full.boundaries.beta.shape == (1, 2, 2)
    assert all(
        torch.isfinite(value).all()
        for value in full.boundaries.__dict__.values()
    )


def test_official_kda_recurrence_decays_key_axis_before_delta_update() -> None:
    cfg = _config()
    weights = _weights()
    token = _tokens()[:, :1]
    state = OfficialKdaState(
        conv_q=torch.tensor(
            [[[0.25, -0.5, 0.75, -1.0], [0.5, 0.25, -0.25, 0.75]]],
            dtype=torch.bfloat16,
        ),
        conv_k=torch.tensor(
            [[[-0.5, 0.25, 0.5, -0.75], [0.25, -0.5, 0.75, 0.5]]],
            dtype=torch.bfloat16,
        ),
        conv_v=torch.tensor(
            [[[0.75, -0.25, 0.5, 0.25], [-0.5, 0.75, -0.25, 0.5]]],
            dtype=torch.bfloat16,
        ),
        recurrent_v_first=(
            torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 32.0
        ),
    )
    original = tuple(value.clone() for value in state.__dict__.values())

    result = official_kda(token, weights, state, cfg)

    q = result.boundaries.q[:, 0].float()
    k = result.boundaries.k[:, 0].float()
    v = result.boundaries.v[:, 0].float()
    alpha = result.boundaries.log_decay[:, 0].float().exp()
    beta = result.boundaries.beta[:, 0].float()
    state_kv = state.recurrent_v_first.transpose(-1, -2)
    decayed = alpha.unsqueeze(-1) * state_kv
    prediction = torch.einsum("bhk,bhkv->bhv", k, decayed)
    delta = (v - prediction) * beta.unsqueeze(-1)
    expected_kv = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
    expected_output = torch.einsum("bhk,bhkv->bhv", q, expected_kv)

    torch.testing.assert_close(
        result.state.recurrent_v_first,
        expected_kv.transpose(-1, -2),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        result.boundaries.recurrent_output[:, 0],
        expected_output,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    for before, after in zip(original, state.__dict__.values()):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize(
    "weights",
    (
        replace(_weights(), a_log=torch.zeros(2, dtype=torch.bfloat16)),
        replace(_weights(), a_log=torch.zeros(3, dtype=torch.float32)),
        replace(_weights(), q_proj=torch.eye(3, dtype=torch.bfloat16)),
        replace(_weights(), q_proj=object()),
    ),
)
def test_official_kda_rejects_dtype_or_shape_drift(weights: OfficialKdaWeights) -> None:
    cfg = _config()
    tokens = _tokens()
    state = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)

    with pytest.raises(ValueError, match="invalid official KDA"):
        official_kda(tokens, weights, state, cfg)


def test_official_kda_rejects_k_first_labeled_state_shape() -> None:
    cfg = _config()
    tokens = _tokens()
    zero = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)
    bad = replace(zero, recurrent_v_first=torch.zeros((1, 2, 3, 2)))

    with pytest.raises(ValueError, match="invalid official KDA"):
        official_kda(tokens, _weights(), bad, cfg)


def test_official_kda_rejects_wrong_convolution_history_width() -> None:
    cfg = _config()
    tokens = _tokens()
    zero = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)
    bad = replace(zero, conv_q=torch.zeros((1, 3, 4), dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="invalid official KDA"):
        official_kda(tokens, _weights(), bad, cfg)


def test_official_kda_rejects_non_finite_weight() -> None:
    cfg = _config()
    tokens = _tokens()
    zero = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)
    bad = replace(_weights(), dt_bias=torch.tensor([0.0, float("nan"), 0.0, 0.0]))

    with pytest.raises(ValueError, match="invalid official KDA"):
        official_kda(tokens, bad, zero, cfg)


def test_official_kda_rejects_empty_sequence() -> None:
    cfg = _config()
    tokens = _tokens()[:, :0]
    zero = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)

    with pytest.raises(ValueError, match="invalid official KDA"):
        official_kda(tokens, _weights(), zero, cfg)
def test_official_kda_projection_is_sequence_partition_invariant() -> None:
    torch.manual_seed(20260811)
    hidden = torch.randn(1, 2, 2_048, dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(1_024, 2_048, dtype=torch.float32).to(torch.bfloat16)

    full = _project(hidden, weight)
    incremental = torch.cat(
        tuple(_project(hidden[:, index : index + 1], weight) for index in range(2)),
        dim=1,
    )

    assert torch.equal(full, incremental)


def test_official_kda_accepts_matvec_projection_weights() -> None:
    class MatvecWeight:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor
            self.shape = tensor.shape
            self.dtype = tensor.dtype
            self.device = tensor.device

        def matvec(self, value: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(
                value.reshape(1, -1), self.tensor
            ).reshape(-1)

    cfg = _config()
    tokens = _tokens()
    weights = _weights()
    zero = zero_official_kda_state(cfg, batch_size=1, device=tokens.device)
    expected = official_kda(tokens, weights, zero, cfg)
    actual = official_kda(
        tokens,
        replace(weights, q_proj=MatvecWeight(weights.q_proj)),
        zero,
        cfg,
    )

    assert torch.equal(actual.output, expected.output)
    assert torch.equal(actual.state.conv_q, expected.state.conv_q)
    assert torch.equal(
        actual.state.recurrent_v_first, expected.state.recurrent_v_first
    )
