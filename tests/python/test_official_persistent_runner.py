# 공식 K3X 그래프 단계의 상주형 호출 계약을 검증합니다.
from __future__ import annotations

from tools import (
    run_official_head,
    run_official_layer0,
    run_official_layer1,
    run_official_layer3,
)


def test_official_stages_expose_callable_run_entrypoints() -> None:
    for module in (
        run_official_layer0,
        run_official_layer1,
        run_official_layer3,
        run_official_head,
    ):
        assert callable(module.run)
