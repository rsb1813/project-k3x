# 공식 Kimi K3 다중 토큰 디코드 상태를 한 프로세스 안에서 유지합니다.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from k3x_converter.format import K3XError


@dataclass
class OfficialInMemoryState:
    attention_states: dict[int, Any] = field(default_factory=dict)
    generated_tokens: list[int] = field(default_factory=list)
    generated_logits: list[float | None] = field(default_factory=list)
    current_token_id: int | None = None
    completed_layer: int = -1
    hidden: torch.Tensor | None = None
    block_sources: tuple[torch.Tensor, ...] = ()
    last_generated_logit: float | None = None

    def begin_token(self, token_id: int) -> None:
        if self.current_token_id is not None:
            raise K3XError("OFFICIAL_IN_MEMORY_TOKEN_ACTIVE")
        self.current_token_id = token_id
        self.completed_layer = -1
        self.hidden = None
        self.block_sources = ()
        self.last_generated_logit = None

    def attention_state(self, layer_id: int) -> Any | None:
        return self.attention_states.get(layer_id)

    def require_layer_input(
        self, layer_id: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if (
            self.current_token_id is None
            or self.completed_layer != layer_id - 1
            or self.hidden is None
            or not self.block_sources
        ):
            raise K3XError("OFFICIAL_IN_MEMORY_LAYER_SEQUENCE")
        return self.hidden, self.block_sources

    def finish_layer(
        self,
        layer_id: int,
        input_hidden: torch.Tensor,
        output_hidden: torch.Tensor,
        attention_state: Any,
        *,
        block_write: bool,
    ) -> None:
        if self.current_token_id is None or self.completed_layer != layer_id - 1:
            raise K3XError("OFFICIAL_IN_MEMORY_LAYER_SEQUENCE")
        if layer_id == 0:
            if not block_write:
                raise K3XError("OFFICIAL_IN_MEMORY_BLOCK_SEQUENCE")
            self.block_sources = (input_hidden,)
        elif block_write:
            self.block_sources = (*self.block_sources, input_hidden)
        self.attention_states[layer_id] = attention_state
        self.hidden = output_hidden
        self.completed_layer = layer_id

    def finish_head(self, token_id: int, logit: float | None = None) -> None:
        if self.current_token_id is None or self.completed_layer != 92:
            raise K3XError("OFFICIAL_IN_MEMORY_HEAD_SEQUENCE")
        self.generated_tokens.append(token_id)
        self.generated_logits.append(logit)
        self.last_generated_logit = logit
        self.current_token_id = None
