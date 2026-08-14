# Gated MLA NoPE 계산과 incremental KV state 동등성을 검증합니다.
import pytest
import torch

from k3x_ref.config import SyntheticK3Config
from k3x_ref.mla import MLAWeights, empty_mla_state, mla_decode, mla_prefill


def _weights(cfg: SyntheticK3Config) -> MLAWeights:
    generator = torch.Generator().manual_seed(20260809)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * 0.05

    query_width = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    return MLAWeights(
        q_a_proj=rand(cfg.q_lora_rank, cfg.hidden_size),
        q_a_norm=torch.linspace(0.9, 1.1, cfg.q_lora_rank),
        q_b_proj=rand(cfg.mla_heads * query_width, cfg.q_lora_rank),
        kv_a_proj=rand(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        kv_a_norm=torch.linspace(0.9, 1.1, cfg.kv_lora_rank),
        kv_b_proj=rand(
            cfg.mla_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            cfg.kv_lora_rank,
        ),
        g_proj=rand(cfg.mla_heads * cfg.v_head_dim, cfg.hidden_size),
        o_proj=rand(cfg.hidden_size, cfg.mla_heads * cfg.v_head_dim),
    )


@pytest.mark.parametrize("length", [1, 2, 5])
def test_mla_prefill_matches_incremental_decode(length: int) -> None:
    cfg = SyntheticK3Config.default()
    weights = _weights(cfg)
    x = torch.arange(length * cfg.hidden_size, dtype=torch.float32).reshape(
        1, length, cfg.hidden_size
    ) / 80.0

    full_output, full_state = mla_prefill(x, weights, None, cfg)
    state = empty_mla_state(1, cfg, x.dtype, x.device)
    pieces = []
    for index in range(length):
        output, state = mla_decode(x[:, index : index + 1], weights, state, cfg)
        pieces.append(output)

    assert torch.equal(full_output, torch.cat(pieces, dim=1))
    assert torch.equal(full_state.keys, state.keys)
    assert torch.equal(full_state.values, state.values)
    assert torch.equal(full_state.shared_keys, state.shared_keys)
    assert full_state.length == state.length == length


def test_mla_extra_key_subspace_is_position_independent() -> None:
    cfg = SyntheticK3Config.default()
    weights = _weights(cfg)
    x = torch.ones((1, 1, cfg.hidden_size))
    first, _ = mla_decode(x, weights, empty_mla_state(1, cfg, x.dtype, x.device), cfg)
    second, _ = mla_decode(x, weights, empty_mla_state(1, cfg, x.dtype, x.device), cfg)
    assert torch.equal(first, second)


def test_mla_decode_preserves_bfloat16_projection_contract() -> None:
    cfg = SyntheticK3Config.default()
    weights = MLAWeights(
        *(tensor.to(torch.bfloat16) for tensor in _weights(cfg).__dict__.values())
    )
    x = torch.ones((1, 1, cfg.hidden_size), dtype=torch.bfloat16)

    output, state = mla_decode(
        x, weights, empty_mla_state(1, cfg, x.dtype, x.device), cfg
    )

    assert output.dtype == torch.bfloat16
    assert state.keys.dtype == torch.bfloat16


def test_mla_decode_accepts_matvec_capable_projection_weights() -> None:
    cfg = SyntheticK3Config.default()
    tensor_weights = _weights(cfg)

    class MatvecWeight:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor
            self.shape = tensor.shape
            self.dtype = tensor.dtype

        def matvec(self, value: torch.Tensor) -> torch.Tensor:
            return self.tensor @ value

    packed_weights = MLAWeights(
        *(
            value if value.ndim == 1 else MatvecWeight(value)
            for value in tensor_weights.__dict__.values()
        )
    )
    hidden = torch.ones((1, 1, cfg.hidden_size))
    empty = empty_mla_state(1, cfg, hidden.dtype, hidden.device)

    expected, expected_state = mla_decode(hidden, tensor_weights, empty, cfg)
    actual, actual_state = mla_decode(hidden, packed_weights, empty, cfg)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_state.keys, expected_state.keys, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        actual_state.values, expected_state.values, atol=1e-5, rtol=1e-5
    )
