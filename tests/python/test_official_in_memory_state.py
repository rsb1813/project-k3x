# 공식 다중 토큰 디코드의 인메모리 상태 생명주기를 검증합니다.
from __future__ import annotations

import pytest
import torch

from k3x_converter.format import K3XError
from tools.official_in_memory_state import OfficialInMemoryState


def test_attention_persists_while_token_block_bank_resets() -> None:
    state = OfficialInMemoryState()
    embedding = torch.tensor([1.0])
    layer0 = torch.tensor([2.0])
    layer1 = torch.tensor([3.0])
    kda0 = object()
    kda1 = object()

    state.begin_token(1)
    assert state.attention_state(0) is None
    state.finish_layer(0, embedding, layer0, kda0, block_write=True)
    state.finish_layer(1, layer0, layer1, kda1, block_write=False)
    prior = layer1
    for layer_id in range(2, 93):
        current = torch.tensor([float(layer_id + 2)])
        state.finish_layer(
            layer_id, prior, current, object(), block_write=False
        )
        prior = current
    state.finish_head(9)

    state.begin_token(9)
    assert state.attention_state(0) is kda0
    assert state.attention_state(1) is kda1
    assert state.block_sources == ()
    assert state.hidden is None
    assert state.generated_tokens == [9]

    next_embedding = torch.tensor([4.0])
    next_hidden = torch.tensor([5.0])
    state.finish_layer(0, next_embedding, next_hidden, object(), block_write=True)
    assert len(state.block_sources) == 1
    assert torch.equal(state.block_sources[0], next_embedding)


def test_state_rejects_nonsequential_layer_and_early_head() -> None:
    state = OfficialInMemoryState()
    state.begin_token(1)
    with pytest.raises(K3XError, match="OFFICIAL_IN_MEMORY_LAYER_SEQUENCE"):
        state.finish_layer(
            1, torch.zeros(1), torch.zeros(1), object(), block_write=False
        )
    with pytest.raises(K3XError, match="OFFICIAL_IN_MEMORY_HEAD_SEQUENCE"):
        state.finish_head(2)
